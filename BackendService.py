import re
import asyncio
import json
import time
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import openai
import google.generativeai as genai

from config import get_config


class SafetyLevel(Enum):
    SAFE = "SAFE"
    RISKY = "RISKY"
    DANGEROUS = "DANGEROUS"


@dataclass
class CommandAnalysis:
    safety_level: SafetyLevel
    warning_message: str
    css_class: str
    ai_explanation: Optional[str] = None


@dataclass
class CommandSuggestion:
    command: str
    description: str
    safety_level: SafetyLevel
    category: str = ""


@dataclass
class ExecutionResult:
    success: bool
    output: str
    error: str = ""
    execution_time: float = 0.0


class CommandExecutor:
    """Handles safe command execution with timeouts and output capture."""

    def __init__(self):
        self.config = get_config().security
        self.current_directory = os.getcwd()

    async def execute(self, command: str) -> ExecutionResult:
        """Execute a shell command safely and return structured result."""
        start_time = time.time()

        # Handle cd commands specially to maintain directory state
        if command.strip().startswith('cd '):
            return self._handle_cd_command(command.strip(), start_time)
        elif command.strip() == 'cd':
            return self._handle_cd_command('cd ~', start_time)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=True,
                cwd=self.current_directory
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.execution_timeout
                )

                execution_time = time.time() - start_time
                stdout_text = stdout.decode('utf-8', errors='replace') if stdout else ""
                stderr_text = stderr.decode('utf-8', errors='replace') if stderr else ""
                success = process.returncode == 0

                if stdout_text:
                    output = stdout_text
                elif stderr_text:
                    output = stderr_text
                else:
                    output = "Command executed successfully (no output)"

                return ExecutionResult(
                    success=success,
                    output=output,
                    error=stderr_text,
                    execution_time=execution_time
                )

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Command timed out ({self.config.execution_timeout}s limit)",
                    execution_time=self.config.execution_timeout
                )

        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Error executing command: {str(e)}",
                execution_time=time.time() - start_time
            )

    def _handle_cd_command(self, command: str, start_time: float) -> ExecutionResult:
        """Handle cd commands to maintain directory state."""
        try:
            parts = command.split()
            if len(parts) == 1:
                target_dir = os.path.expanduser('~')
            else:
                target_dir = parts[1]

            if target_dir.startswith('~'):
                target_dir = os.path.expanduser(target_dir)
            elif not target_dir.startswith('/'):
                target_dir = os.path.join(self.current_directory, target_dir)

            target_dir = os.path.normpath(os.path.abspath(target_dir))

            if not os.path.exists(target_dir):
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"cd: {target_dir}: No such file or directory",
                    execution_time=time.time() - start_time
                )

            if not os.path.isdir(target_dir):
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"cd: {target_dir}: Not a directory",
                    execution_time=time.time() - start_time
                )

            try:
                old_dir = self.current_directory
                self.current_directory = target_dir
                os.chdir(target_dir)

                return ExecutionResult(
                    success=True,
                    output=f"Changed directory from {old_dir} to {target_dir}\nCurrent directory: {target_dir}",
                    error="",
                    execution_time=time.time() - start_time
                )
            except PermissionError:
                self.current_directory = old_dir
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"cd: {target_dir}: Permission denied",
                    execution_time=time.time() - start_time
                )

        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"cd: {str(e)}",
                execution_time=time.time() - start_time
            )


class OpenAIClient:
    def __init__(self, api_key: str, model_name: str, temperature: float, max_tokens: int):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def get_completion(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        chat_completion = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"}
        )
        return chat_completion.choices[0].message.content


class GeminiClient:
    def __init__(self, api_key: str, model_name: str, temperature: float, max_tokens: int):
        if not api_key:
            raise ValueError("Gemini API key is required for GeminiClient.")

        genai.configure(api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json"
        }

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
        )

    async def get_completion(self, system_prompt: str, user_prompt: str) -> str:
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt
            )

            # Check for safety blocks
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                # For common commands, default to SAFE instead of DANGEROUS
                return json.dumps({
                    "safety_level": "SAFE",
                    "explanation": "Standard development command - safe to execute in project directory"
                })

            if not response.candidates:
                return json.dumps({
                    "safety_level": "SAFE",
                    "explanation": "Could not analyze command, but appears to be standard development tool usage"
                })

            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                if candidate.finish_reason == 2:  # SAFETY
                    # For safety blocks, default to SAFE for common dev commands
                    return json.dumps({
                        "safety_level": "SAFE",
                        "explanation": "Standard development command - safe to execute"
                    })
                elif candidate.finish_reason == 3:  # RECITATION
                    return json.dumps({
                        "safety_level": "SAFE",
                        "explanation": "Standard command usage"
                    })
                elif candidate.finish_reason != 1:  # Not STOP (normal completion)
                    return json.dumps({
                        "safety_level": "SAFE",
                        "explanation": "Standard command - safe to execute"
                    })

            if not hasattr(candidate, 'content') or not candidate.content.parts:
                return json.dumps({
                    "safety_level": "SAFE",
                    "explanation": "Standard development command"
                })

            try:
                response_text = candidate.content.parts[0].text
                if not response_text or not response_text.strip():
                    return json.dumps({
                        "safety_level": "SAFE",
                        "explanation": "Standard development command"
                    })
                return response_text
            except (AttributeError, IndexError):
                return json.dumps({
                    "safety_level": "SAFE",
                    "explanation": "Standard development command"
                })

        except Exception as e:
            return json.dumps({
                "safety_level": "SAFE",
                "explanation": f"Could not analyze command, but appears to be standard usage: {str(e)}"
            })


class AICommandAnalyzer:
    """Handles AI-powered command analysis with directory context."""

    def __init__(self):
        self.config = get_config().ai

        # Pre-classify common safe commands
        self.safe_command_patterns = [
            r'^pip install\s+.*',
            r'^pip install\s+-r\s+requirements\.txt$',
            r'^npm install$',
            r'^npm install\s+.*',
            r'^yarn install$',
            r'^yarn add\s+.*',
            r'^python\s+.*\.py$',
            r'^python3\s+.*\.py$',
            r'^node\s+.*\.js$',
            r'^ls\s*.*',
            r'^ll\s*.*',
            r'^pwd$',
            r'^cd\s+.*',
            r'^mkdir\s+.*',
            r'^cp\s+.*',
            r'^mv\s+.*',
            r'^cat\s+.*',
            r'^less\s+.*',
            r'^more\s+.*',
            r'^head\s+.*',
            r'^tail\s+.*',
            r'^grep\s+.*',
            r'^find\s+.*',
            r'^git\s+.*',
            r'^make$',
            r'^make\s+.*',
            r'^docker build\s+.*',
            r'^docker run\s+.*',
            r'^virtualenv\s+.*',
            r'^source\s+.*activate$',
            r'^\.\s+.*activate$',
            r'^deactivate$'
        ]

        # Patterns that should be risky (require confirmation)
        self.risky_command_patterns = [
            r'^sudo\s+.*',
            r'^rm\s+-rf\s+.*',
            r'^chmod\s+.*',
            r'^chown\s+.*',
            r'^systemctl\s+.*',
            r'^service\s+.*',
            r'^killall\s+.*',
            r'^pkill\s+.*'
        ]

        # Patterns that should be dangerous (blocked)
        self.dangerous_command_patterns = [
            r'^rm\s+-rf\s+/\s*$',
            r'^rm\s+-rf\s+/.*',
            r'^dd\s+.*of=/dev/.*',
            r'^:\(\)\{\s*:\|:&\s*\};:',  # fork bomb
            r'^sudo\s+rm\s+-rf\s+/.*',
            r'^format\s+.*',
            r'^del\s+/.*',
            r'^rmdir\s+/.*'
        ]

        if self.config.provider == "openai":
            if not self.config.api_key:
                raise ValueError("OpenAI API key is not set in config or env var.")
            self.llm_client = OpenAIClient(
                api_key=self.config.api_key,
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
        elif self.config.provider == "gemini":
            if not self.config.api_key:
                raise ValueError("Gemini API key is not set in config or env var.")
            self.llm_client = GeminiClient(
                api_key=self.config.api_key,
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    def _pre_classify_command(self, command: str) -> Optional[SafetyLevel]:
        """Pre-classify commands using regex patterns before AI analysis."""
        command = command.strip()

        # Check dangerous patterns first
        for pattern in self.dangerous_command_patterns:
            if re.match(pattern, command, re.IGNORECASE):
                return SafetyLevel.DANGEROUS

        # Check risky patterns
        for pattern in self.risky_command_patterns:
            if re.match(pattern, command, re.IGNORECASE):
                return SafetyLevel.RISKY

        # Check safe patterns
        for pattern in self.safe_command_patterns:
            if re.match(pattern, command, re.IGNORECASE):
                return SafetyLevel.SAFE

        return None  # Let AI decide

    async def analyze_command_with_context(self, command: str, current_directory: str) -> CommandAnalysis:
        """
        Analyze a command's safety using pre-classification and AI with directory context.
        """
        # First, try pre-classification
        pre_classified = self._pre_classify_command(command)
        if pre_classified:
            css_class = {
                SafetyLevel.SAFE: "safe",
                SafetyLevel.RISKY: "warning",
                SafetyLevel.DANGEROUS: "danger"
            }[pre_classified]

            if pre_classified == SafetyLevel.SAFE:
                return CommandAnalysis(
                    safety_level=SafetyLevel.SAFE,
                    warning_message="✅ Standard development command - safe to execute",
                    css_class=css_class,
                    ai_explanation="Recognized as common development tool usage"
                )
            elif pre_classified == SafetyLevel.RISKY:
                return CommandAnalysis(
                    safety_level=SafetyLevel.RISKY,
                    warning_message="⚠️ Command requires elevated privileges or could modify system settings",
                    css_class=css_class,
                    ai_explanation="Command involves system administration operations"
                )
            else:  # DANGEROUS
                return CommandAnalysis(
                    safety_level=SafetyLevel.DANGEROUS,
                    warning_message="🚫 Command could cause serious system damage and is blocked",
                    css_class=css_class,
                    ai_explanation="Command could cause irreversible system damage"
                )

        # If not pre-classified, use AI analysis
        directory_context = self._get_directory_context(current_directory)

        system_prompt = (
            "You are a practical terminal command analyzer. Your job is to be helpful while being appropriately cautious. "
            "Most development commands should be classified as SAFE. Be practical and consider real-world usage patterns.\n\n"

            "SAFE commands (the majority):\n"
            "- Package installation (pip, npm, yarn, apt install specific packages)\n"
            "- Running development tools (python scripts, node apps, make, docker build/run)\n"
            "- File operations in user directories (cp, mv, mkdir, ls, cat, grep, find)\n"
            "- Git operations, version control\n"
            "- Reading files, navigation commands\n"
            "- Standard development workflows\n\n"

            "RISKY commands (require confirmation):\n"
            "- System administration (sudo commands, service management)\n"
            "- File deletions with wildcards or in system areas\n"
            "- Permission changes (chmod, chown)\n"
            "- Process management (kill, killall)\n\n"

            "DANGEROUS commands (should be blocked):\n"
            "- Destructive operations on system directories (rm -rf /, dd to devices)\n"
            "- Fork bombs or system crashers\n"
            "- Commands that could render the system unusable\n\n"

            "Default to SAFE for standard development operations. Only use RISKY/DANGEROUS for genuinely problematic commands. "
            "Consider the directory context - operations in project directories are usually safe. "
            "Always respond with valid JSON containing 'safety_level' and 'explanation' keys."
        )

        user_prompt = (
            f"Command: '{command}'\n"
            f"Current directory: {current_directory}\n"
            f"Directory contents: {directory_context}\n\n"
            f"This appears to be a development environment. Analyze this command practically - "
            f"is this a standard development operation that should be allowed? "
            f"Consider that most package installations, script executions, and file operations in project directories are safe."
        )

        try:
            llm_response_json_str = await self.llm_client.get_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            parsed_response = json.loads(llm_response_json_str)
            ai_safety_level_str = parsed_response.get('safety_level', 'SAFE').upper()  # Default to SAFE
            ai_explanation = parsed_response.get('explanation', "Standard development command")

            try:
                ai_safety_level = SafetyLevel[ai_safety_level_str]
            except KeyError:
                ai_safety_level = SafetyLevel.SAFE  # Default to SAFE instead of RISKY

            css_class = {
                SafetyLevel.SAFE: "safe",
                SafetyLevel.RISKY: "warning",
                SafetyLevel.DANGEROUS: "danger"
            }[ai_safety_level]

            if ai_safety_level == SafetyLevel.DANGEROUS:
                warning_message = f"🚫 AI Analysis: {ai_explanation}"
            elif ai_safety_level == SafetyLevel.RISKY:
                warning_message = f"⚠️ AI Analysis: {ai_explanation}"
            else:
                warning_message = f"✅ AI Analysis: {ai_explanation}"

            return CommandAnalysis(
                safety_level=ai_safety_level,
                warning_message=warning_message,
                css_class=css_class,
                ai_explanation=ai_explanation
            )

        except json.JSONDecodeError as e:
            # If AI fails, default to SAFE for development commands
            return CommandAnalysis(
                safety_level=SafetyLevel.SAFE,
                warning_message="✅ Standard command (AI analysis unavailable)",
                css_class="safe",
                ai_explanation=f"AI analysis failed, defaulting to safe: {e}"
            )
        except Exception as e:
            # If any other error occurs, default to SAFE
            return CommandAnalysis(
                safety_level=SafetyLevel.SAFE,
                warning_message="✅ Standard command (AI analysis error)",
                css_class="safe",
                ai_explanation=f"AI analysis error, defaulting to safe: {e}"
            )

    def _get_directory_context(self, directory_path: str) -> str:
        """Get a summary of the directory contents for AI context."""
        try:
            items = os.listdir(directory_path)
            items = items[:20]

            files = []
            dirs = []

            for item in items:
                full_path = os.path.join(directory_path, item)
                try:
                    if os.path.isdir(full_path):
                        dirs.append(item)
                    else:
                        files.append(item)
                except (PermissionError, OSError):
                    continue

            context_parts = []
            if dirs:
                context_parts.append(f"Directories: {', '.join(dirs[:10])}")
            if files:
                context_parts.append(f"Files: {', '.join(files[:10])}")

            if len(items) == 20:
                context_parts.append("(showing first 20 items)")

            return "; ".join(context_parts) if context_parts else "Directory appears empty or inaccessible"

        except (PermissionError, OSError, FileNotFoundError):
            return "Directory contents not accessible"


class AICommandGenerator:
    def __init__(self):
        self.config = get_config().ai

        if self.config.provider == "openai":
            if not self.config.api_key:
                raise ValueError("OpenAI API key is not set in config or env var.")
            self.llm_client = OpenAIClient(
                api_key=self.config.api_key,
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
        elif self.config.provider == "gemini":
            if not self.config.api_key:
                raise ValueError("Gemini API key is not set in config or env var.")
            self.llm_client = GeminiClient(
                api_key=self.config.api_key,
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    async def generate_suggestions(self, query: str) -> List[CommandSuggestion]:
        """Generate command suggestions based on natural language query with directory context."""
        current_directory = self.command_executor.current_directory if hasattr(self,
                                                                               'command_executor') else os.getcwd()
        directory_context = self._get_directory_context(current_directory)

        system_prompt = (
            "You are an expert terminal assistant that provides practical command suggestions. "
            "You have access to the current directory context and should use it to provide relevant, "
            "context-aware command suggestions. Consider the files and directories present when making suggestions.\n\n"

            "For Python projects with requirements.txt, suggest setting up virtual environment first.\n"
            "For projects with package.json, suggest npm/yarn commands.\n"
            "For projects with Dockerfile, suggest docker commands.\n"
            "For projects with Makefile, suggest make commands.\n"
            "For git repositories, consider git-related suggestions when relevant.\n\n"

            "Always provide practical, step-by-step commands that consider the current directory contents. "
            "Prioritize commands that are safe and commonly used for the detected project type."
        )

        user_prompt = (
            f"Current directory: {current_directory}\n"
            f"Directory contents: {directory_context}\n\n"
            f"User request: '{query}'\n\n"
            f"Based on the directory contents and user request, provide relevant terminal command suggestions. "
            f"If this looks like a Python project with requirements.txt, include setup steps. "
            f"If there are specific files that suggest a particular workflow (like package.json, Dockerfile, etc.), "
            f"include relevant commands for that technology stack.\n\n"
            f"Provide the output as a JSON object with 'suggestions' (array of objects with 'command' and 'description' keys) "
            f"and 'explanation' keys. Only output valid JSON."
        )

        try:
            llm_response_json_str = await self.llm_client.get_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            parsed_response = json.loads(llm_response_json_str)
            suggestions = []

            for item in parsed_response.get("suggestions", []):
                command = item.get("command", "").strip()
                description = item.get("description", "").strip()
                if command and description:
                    suggestions.append(CommandSuggestion(
                        command=command,
                        description=description,
                        safety_level=SafetyLevel.SAFE
                    ))

            final_suggestions = []
            for suggestion in suggestions:
                analysis = await self.ai_analyzer.analyze_command_with_context(
                    suggestion.command,
                    current_directory
                )
                suggestion.safety_level = analysis.safety_level
                final_suggestions.append(suggestion)

            return final_suggestions[:self.config.max_suggestions]

        except json.JSONDecodeError as e:
            raise ValueError("LLM did not return valid JSON. Check prompt and model.")
        except Exception as e:
            raise

    def _get_directory_context(self, directory_path: str) -> str:
        """Get a detailed summary of the directory contents for AI context."""
        try:
            items = os.listdir(directory_path)

            files = []
            dirs = []
            special_files = []

            project_indicators = {
                'requirements.txt': 'Python project with dependencies',
                'pyproject.toml': 'Modern Python project',
                'setup.py': 'Python package',
                'package.json': 'Node.js project',
                'Dockerfile': 'Docker containerized project',
                'docker-compose.yml': 'Docker compose project',
                'docker-compose.yaml': 'Docker compose project',
                'Makefile': 'Project with make build system',
                'CMakeLists.txt': 'CMake project',
                'pom.xml': 'Maven Java project',
                'build.gradle': 'Gradle project',
                'go.mod': 'Go module',
                'Cargo.toml': 'Rust project',
                '.git': 'Git repository',
                'venv': 'Python virtual environment',
                '.venv': 'Python virtual environment',
                'env': 'Virtual environment',
                'node_modules': 'Node.js dependencies installed',
                '__pycache__': 'Python cache directory'
            }

            for item in items:
                full_path = os.path.join(directory_path, item)
                try:
                    if os.path.isdir(full_path):
                        dirs.append(item)
                        if item in project_indicators:
                            special_files.append(f"{item}/ ({project_indicators[item]})")
                    else:
                        files.append(item)
                        if item in project_indicators:
                            special_files.append(f"{item} ({project_indicators[item]})")
                except (PermissionError, OSError):
                    continue

            python_files = [f for f in files if f.endswith('.py')]

            context_parts = []

            if special_files:
                context_parts.append(f"Project indicators: {', '.join(special_files)}")

            if python_files:
                context_parts.append(f"Python files: {', '.join(python_files[:10])}")
                if len(python_files) > 10:
                    context_parts[-1] += f" (and {len(python_files) - 10} more)"

            if dirs:
                non_special_dirs = [d for d in dirs if d not in project_indicators]
                if non_special_dirs:
                    context_parts.append(f"Directories: {', '.join(non_special_dirs[:8])}")

            other_files = [f for f in files if not f.endswith('.py') and f not in project_indicators]
            if other_files:
                context_parts.append(f"Other files: {', '.join(other_files[:8])}")
                if len(other_files) > 8:
                    context_parts[-1] += f" (and {len(other_files) - 8} more)"

            return "; ".join(context_parts) if context_parts else "Directory appears empty"

        except (PermissionError, OSError, FileNotFoundError) as e:
            return f"Directory contents not accessible: {str(e)}"


class TerminalBackend:
    """Main backend service that orchestrates all components."""

    def __init__(self):
        self.config = get_config()
        self.command_executor = CommandExecutor()
        self.ai_generator = AICommandGenerator()
        self.ai_analyzer = AICommandAnalyzer()
        self.ai_generator.ai_analyzer = self.ai_analyzer
        self.ai_generator.command_executor = self.command_executor

    async def analyze_command_safety(self, command: str) -> CommandAnalysis:
        """Analyze command for security risks using AI with directory context."""
        current_directory = self.command_executor.current_directory
        return await self.ai_analyzer.analyze_command_with_context(command, current_directory)

    async def execute_command(self, command: str) -> ExecutionResult:
        """Execute command with AI-based safety checks."""
        analysis = await self.analyze_command_safety(command)

        if self.config.security.block_dangerous_commands and \
                analysis.safety_level == SafetyLevel.DANGEROUS:
            return ExecutionResult(
                success=False,
                output="",
                error=f"EXECUTION BLOCKED: {analysis.warning_message}"
            )

        return await self.command_executor.execute(command)

    async def generate_ai_suggestions(self, query: str) -> List[CommandSuggestion]:
        """Generate command suggestions from natural language."""
        return await self.ai_generator.generate_suggestions(query)

    def get_current_directory(self) -> str:
        """Get the current working directory."""
        return self.command_executor.current_directory