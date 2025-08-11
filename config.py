import os
import json
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SecurityConfig:
    """Security-related configuration settings."""
    execution_timeout: float = 10.0
    block_dangerous_commands: bool = True
    require_confirmation_for_risky: bool = True
    custom_dangerous_patterns: List[str] = field(default_factory=list)
    custom_risky_patterns: List[str] = field(default_factory=list)
    log_executed_commands: bool = True


@dataclass
class AIConfig:
    """AI/LLM-related configuration settings."""
    provider: str = "gemini"
    api_key: Optional[str] = None
    model_name: str = "gemini-2.5-flash"
    max_suggestions: int = 6
    temperature: float = 0.3
    max_tokens: int = 1000
    system_prompt: str = """You are a helpful terminal assistant. Generate safe, practical command suggestions based on user queries. Always prioritize security and explain what each command does."""


@dataclass
class UIConfig:
    """User interface configuration settings."""
    theme: str = "dark"
    show_progress_bars: bool = True
    show_safety_indicators: bool = True
    default_tab: str = "command_mode"
    font_size_multiplier: float = 1.0


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    enabled: bool = True
    level: str = "INFO"
    log_file: Optional[str] = "terminal_assistant.log"
    max_file_size_mb: int = 10
    backup_count: int = 3


@dataclass
class AppConfig:
    """Main application configuration."""
    security: SecurityConfig = field(default_factory=SecurityConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    app_name: str = "Terminal Assistant"
    version: str = "1.0.0"

    @classmethod
    def load_from_file(cls, config_path: str = "config.json") -> "AppConfig":
        """Load configuration from JSON file."""
        if not os.path.exists(config_path):
            config = cls()
            config.save_to_file(config_path)
            return config

        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            security_data = config_data.get('security', {})
            ai_data = config_data.get('ai', {})
            ui_data = config_data.get('ui', {})
            logging_data = config_data.get('logging', {})

            return cls(
                security=SecurityConfig(**security_data),
                ai=AIConfig(**ai_data),
                ui=UIConfig(**ui_data),
                logging=LoggingConfig(**logging_data),
                app_name=config_data.get('app_name', "Terminal Assistant"),
                version=config_data.get('version', "1.0.0")
            )

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"Error loading config from {config_path}: {e}")
            print("Using default configuration")
            return cls()

    def save_to_file(self, config_path: str = "config.json") -> None:
        """Save configuration to JSON file."""
        config_data = {
            'security': {
                'execution_timeout': self.security.execution_timeout,
                'block_dangerous_commands': self.security.block_dangerous_commands,
                'require_confirmation_for_risky': self.security.require_confirmation_for_risky,
                'custom_dangerous_patterns': self.security.custom_dangerous_patterns,
                'custom_risky_patterns': self.security.custom_risky_patterns,
                'log_executed_commands': self.security.log_executed_commands
            },
            'ai': {
                'provider': self.ai.provider,
                'api_key': self.ai.api_key,
                'model_name': self.ai.model_name,
                'max_suggestions': self.ai.max_suggestions,
                'temperature': self.ai.temperature,
                'max_tokens': self.ai.max_tokens,
                'system_prompt': self.ai.system_prompt
            },
            'ui': {
                'theme': self.ui.theme,
                'show_progress_bars': self.ui.show_progress_bars,
                'show_safety_indicators': self.ui.show_safety_indicators,
                'default_tab': self.ui.default_tab,
                'font_size_multiplier': self.ui.font_size_multiplier
            },
            'logging': {
                'enabled': self.logging.enabled,
                'level': self.logging.level,
                'log_file': self.logging.log_file,
                'max_file_size_mb': self.logging.max_file_size_mb,
                'backup_count': self.logging.backup_count
            },
            'app_name': self.app_name,
            'version': self.version
        }

        try:
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            print(f"Error saving config to {config_path}: {e}")

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        config = cls()

        if timeout := os.getenv('TERMINAL_ASSISTANT_TIMEOUT'):
            config.security.execution_timeout = float(timeout)

        if block_dangerous := os.getenv('TERMINAL_ASSISTANT_BLOCK_DANGEROUS'):
            config.security.block_dangerous_commands = block_dangerous.lower() in ('true', '1', 'yes')

        if provider := os.getenv('TERMINAL_ASSISTANT_AI_PROVIDER'):
            config.ai.provider = provider

        if api_key := os.getenv('TERMINAL_ASSISTANT_API_KEY'):
            config.ai.api_key = api_key

        if model := os.getenv('TERMINAL_ASSISTANT_MODEL'):
            config.ai.model_name = model

        if theme := os.getenv('TERMINAL_ASSISTANT_THEME'):
            config.ui.theme = theme

        if default_tab := os.getenv('TERMINAL_ASSISTANT_DEFAULT_TAB'):
            config.ui.default_tab = default_tab

        if log_level := os.getenv('TERMINAL_ASSISTANT_LOG_LEVEL'):
            config.logging.level = log_level.upper()

        if log_file := os.getenv('TERMINAL_ASSISTANT_LOG_FILE'):
            config.logging.log_file = log_file if log_file != 'none' else None

        return config


# Global configuration instance
_config_instance: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig.load_from_file()

        env_config = AppConfig.from_env()
        if env_config.ai.api_key:
            _config_instance.ai.api_key = env_config.ai.api_key
        if env_config.ai.provider != "mock":
            _config_instance.ai.provider = env_config.ai.provider
        if env_config.ai.model_name != "gpt-3.5-turbo":
            _config_instance.ai.model_name = env_config.ai.model_name

    return _config_instance


def reload_config() -> AppConfig:
    """Reload configuration from file."""
    global _config_instance
    _config_instance = None
    return get_config()