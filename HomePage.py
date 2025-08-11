from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal
from textual.events import Key
from textual.reactive import var
from textual.widgets import (
    Header, Footer, TabbedContent, TabPane, TextArea, Static,
    Button, ListView, ListItem, Label, ProgressBar
)

from BackendService import TerminalBackend, SafetyLevel


class TerminalAssistantApp(App):
    """Main terminal assistant application with separated frontend/backend."""

    CSS = """
    .input-label {
        color: $text-muted;
        margin: 1 0 0 0;
        text-style: italic;
    }

    Screen {
        background: $surface;
    }

    #command_text_input, #ai_text_input {
        height: 4;
        border: solid $primary;
        margin: 1 0;
    }

    #command_text_output {
        height: 5fr;
        border: solid $secondary;
        background: $panel;
        margin: 1 0;
    }

    .danger {
        color: $error;
        text-style: bold;
    }

    .warning {
        color: $warning;
        text-style: bold;
    }

    .safe {
        color: $success;
    }

    .action-button {
        margin: 0 1;
        min-width: 15;
    }

    .clear-button {
        margin: 0 1;
        min-width: 10;
    }

    ListView {
        height: 25;
        border: solid $primary;
        margin: 1 0;
    }

    ListItem {
        padding: 0 1;
    }

    ListItem:hover {
        background: $primary 20%;
    }

    .status-bar {
        background: $panel;
        height: 3;
        border: solid $secondary;
        padding: 1;
        margin: 1 0;
    }

    ProgressBar {
        margin: 1 0;
    }

    .suggestion-item {
        padding: 1;
        margin: 0 0 1 0;
    }
    """

    # Reactive variables
    is_executing = var(False)
    is_generating = var(False)

    def __init__(self):
        super().__init__()
        self.backend = TerminalBackend()
        self._confirmation_pending = False
        self._pending_command = ""

    def compose(self) -> ComposeResult:
        """Compose the main UI layout."""
        yield Header()
        yield Footer()

        with TabbedContent():
            # Command Mode Tab
            with TabPane("Command Mode", id="command_mode"):
                with Vertical():
                    yield Label("💻 Direct Terminal Command Execution")

                    self.current_dir_label = Label(f"📁 Current directory: {self.backend.get_current_directory()}",classes="input-label")
                    yield self.current_dir_label
                    yield Label("Enter terminal command:", classes="input-label")
                    self.command_input = TextArea(id="command_text_input")
                    yield self.command_input

                    with Horizontal():
                        yield Button("🚀 Execute", id="execute_cmd", variant="primary", classes="action-button")
                        yield Button("🔍 Analyze", id="analyze_cmd", variant="default", classes="clear-button")
                        yield Button("🗑️ Clear", id="clear_cmd", variant="default", classes="clear-button")

                    self.command_output = TextArea(
                        "Command output will appear here...",
                        id="command_text_output",
                        read_only=True,
                        show_line_numbers=False
                    )
                    yield self.command_output

            # AI Mode Tab
            with TabPane("AI Assistant", id="ai_mode"):
                with Vertical():
                    yield Label("🤖 AI-Powered Command Assistant")

                    yield Label("Describe what you want to do:", classes="input-label")
                    self.ai_input = TextArea(id="ai_text_input")
                    yield self.ai_input

                    with Horizontal():
                        yield Button("✨ Get Suggestions", id="get_suggestions", variant="primary",classes="action-button")
                        yield Button("🗑️ Clear", id="clear_ai", variant="default", classes="clear-button")

                    self.ai_progress = ProgressBar(show_eta=False, show_percentage=False)
                    self.ai_progress.display = False
                    yield self.ai_progress

                    yield Label("💡 Command Suggestions:")
                    self.suggestions_list = ListView(id="suggestions_list")
                    yield self.suggestions_list

                    self.ai_output = Static(
                        "[dim]AI suggestions and explanations will appear here...[/dim]",
                        id="ai_text_output"
                    )
                    yield self.ai_output

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id

        if button_id == "execute_cmd":
            await self._handle_command_execution()
        elif button_id == "analyze_cmd":
            await self._handle_command_analysis()
        elif button_id == "clear_cmd":
            self._clear_command_mode()
        elif button_id == "get_suggestions":
            self._handle_ai_suggestions()
        elif button_id == "clear_ai":
            self._clear_ai_mode()

    async def on_key(self, event: Key) -> None:
        """Handle keyboard events."""
        if event.key == "enter":
            if self.command_input.has_focus:
                await self._handle_command_execution()
                event.prevent_default()
            elif self.ai_input.has_focus:
                self._handle_ai_suggestions()
                event.prevent_default()
        elif event.key == "ctrl+c":
            if self.command_input.has_focus:
                self.command_input.clear()
            elif self.ai_input.has_focus:
                self.ai_input.clear()
        elif event.key == "escape":
            self._confirmation_pending = False
            self._pending_command = ""

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle AI suggestion selection."""
        if hasattr(event.item, 'command'):
            command = event.item.command
            self.command_input.text = command
            self.notify(f"📋 Copied to Command Mode: {command}", timeout=3)

            tabbed_content = self.query_one(TabbedContent)
            tabbed_content.active = "command_mode"

    async def _handle_command_execution(self) -> None:
        """Handle command execution with AI-based safety checks."""
        command = self.command_input.text.strip()
        if not command:
            self.notify("⚠️ Please enter a command", severity="warning")
            return

        self.command_output.text = "🤖 Analyzing command safety with AI..."

        try:
            analysis = await self.backend.analyze_command_safety(command)
        except Exception as e:
            self.command_output.text = (
                f"[bold red]AI ANALYSIS FAILED[/bold red]\n\n"
                f"Error: {str(e)}\n\n"
                f"Cannot execute command without safety analysis."
            )
            self.notify("🚫 AI analysis failed - execution blocked", severity="error")
            return

        if analysis.safety_level == SafetyLevel.DANGEROUS:
            self.command_output.text = (
                f"[bold red]DANGEROUS COMMAND BLOCKED[/bold red]\n\n"
                f"{analysis.warning_message}\n\n"
                f"This command is too risky to execute and has been blocked for your safety."
            )
            self.notify("🚫 Dangerous command blocked!", severity="error")
            return

        if analysis.safety_level == SafetyLevel.RISKY:
            if not self._confirmation_pending or self._pending_command != command:
                self._confirmation_pending = True
                self._pending_command = command
                self.command_output.text = (
                    f"[bold yellow]RISKY COMMAND DETECTED[/bold yellow]\n\n"
                    f"{analysis.warning_message}\n\n"
                    f"Press 'Execute' again to confirm execution, or 'Clear' to cancel."
                )
                self.notify("⚠️ Risky command - press Execute again to confirm", severity="warning")
                return

        self._confirmation_pending = False
        self._pending_command = ""
        self._execute_command(command, analysis)

    @work(exclusive=True)
    async def _execute_command(self, command: str, analysis) -> None:
        """Execute command with progress indication."""
        self.is_executing = True

        try:
            result = await self.backend.execute_command(command)

            if result.success:
                output_text = (
                    f"Command: {command}\n"
                    f"Status: ✅ Success\n"
                    f"Execution time: {result.execution_time:.2f}s\n"
                    f"Safety: {analysis.warning_message}\n\n"
                    f"{'=' * 60}\nOUTPUT:\n{'=' * 60}\n\n"
                    f"{result.output or 'No output produced'}"
                )
                self.command_output.text = output_text
                self.current_dir_label.update(f"📁 Current directory: {self.backend.get_current_directory()}")
            else:
                self.command_output.text = (
                    f"Command: {command}\n"
                    f"Status: ❌ Failed\n"
                    f"Error: {result.error}"
                )

        except Exception as e:
            self.command_output.text = f"Exception: {str(e)}"
        finally:
            self.is_executing = False

    async def _handle_command_analysis(self) -> None:
        """Analyze command safety using AI with directory context."""
        command = self.command_input.text.strip()
        if not command:
            self.notify("⚠️ Please enter a command to analyze", severity="warning")
            return

        # Show progress indication
        self.command_output.text = "🤖 Analyzing command with AI..."

        try:
            analysis = await self.backend.analyze_command_safety(command)

            analysis_text = (
                f"Command: [bold]{command}[/bold]\n\n"
                f"Current Directory: [dim]{self.backend.get_current_directory()}[/dim]\n\n"
                f"Safety Level: [{analysis.css_class}]{analysis.safety_level.value}[/{analysis.css_class}]\n\n"
                f"Analysis: {analysis.warning_message}\n\n"
            )

            if analysis.safety_level == SafetyLevel.DANGEROUS:
                analysis_text += "[bold red]🚫 This command is blocked and will not execute.[/bold red]\n"
            elif analysis.safety_level == SafetyLevel.RISKY:
                analysis_text += "[bold yellow]⚠️ This command requires confirmation before execution.[/bold yellow]\n"
            else:
                analysis_text += "[bold green]✅ This command is safe to execute.[/bold green]\n"

            if analysis.ai_explanation:
                analysis_text += f"\n[dim]AI provided additional context in the analysis above.[/dim]"

            analysis_text += "\n\nUse the Execute button to run the command (if safe)."
            self.command_output.text = analysis_text

        except Exception as e:
            # If AI fails, show error message
            error_text = (
                f"Command: [bold]{command}[/bold]\n\n"
                f"[bold red]AI Analysis Failed[/bold red]\n"
                f"Error: {str(e)}\n\n"
                "Please check your AI configuration or try again."
            )
            self.command_output.text = error_text

    def _clear_command_mode(self) -> None:
        """Clear command mode interface."""
        self.command_input.clear()
        self.command_output.text = "Command output will appear here..."
        self._confirmation_pending = False
        self._pending_command = ""

    @work(exclusive=True)
    async def _handle_ai_suggestions(self) -> None:
        """Handle AI suggestion generation."""
        query = self.ai_input.text.strip()
        if not query:
            self.notify("⚠️ Please enter a description", severity="warning")
            return

        self.is_generating = True
        self.ai_progress.display = True

        try:
            self.ai_output.update("🤖 Generating command suggestions...")
            suggestions = await self.backend.generate_ai_suggestions(query)
            await self.suggestions_list.clear()

            if suggestions:
                for suggestion in suggestions:
                    safety_icon = {
                        SafetyLevel.SAFE: "✅",
                        SafetyLevel.RISKY: "⚠️",
                        SafetyLevel.DANGEROUS: "🚫"
                    }.get(suggestion.safety_level, "❓")

                    analysis = await self.backend.analyze_command_safety(suggestion.command)
                    item_text = f"{safety_icon} [{analysis.css_class}]{suggestion.command}[/{analysis.css_class}]\n   {suggestion.description}"

                    list_item = ListItem(Label(item_text, classes="suggestion-item"))
                    list_item.command = suggestion.command
                    await self.suggestions_list.append(list_item)

                self.ai_output.update(
                    f"✨ Generated {len(suggestions)} suggestions for: '[bold]{query}[/bold]'\n\n"
                    f"📋 Click any suggestion to copy it to Command Mode\n"
                    f"Legend: ✅ Safe  ⚠️ Risky  🚫 Dangerous"
                )
            else:
                self.ai_output.update(
                    f"🤔 No suggestions found for: '[bold]{query}[/bold]'\n\n"
                    f"Try rephrasing your request or being more specific."
                )

        except Exception as e:
            self.ai_output.update(f"[bold red]❌ Error generating suggestions[/bold red]\n\nError: {str(e)}")
        finally:
            self.is_generating = False
            self.ai_progress.display = False

    def _clear_ai_mode(self) -> None:
        """Clear AI mode interface."""
        self.ai_input.clear()
        self.suggestions_list.clear()
        self.ai_output.update("[dim]AI suggestions and explanations will appear here...[/dim]")

    def watch_is_executing(self, is_executing: bool) -> None:
        """Update UI state when command execution status changes."""
        try:
            execute_button = self.query_one("#execute_cmd", Button)
            if is_executing:
                execute_button.label = "⏳ Executing..."
                execute_button.disabled = True
            else:
                execute_button.label = "🚀 Execute"
                execute_button.disabled = False
        except:
            pass

    def watch_is_generating(self, is_generating: bool) -> None:
        """Update UI state when AI generation status changes."""
        try:
            suggestions_button = self.query_one("#get_suggestions", Button)
            if is_generating:
                suggestions_button.label = "🤖 Generating..."
                suggestions_button.disabled = True
            else:
                suggestions_button.label = "✨ Get Suggestions"
                suggestions_button.disabled = False
        except:
            pass


def main():
    """Entry point for the application."""
    app = TerminalAssistantApp()
    app.title = "Terminal Assistant"
    app.sub_title = "AI-Powered Command Line Helper"
    app.run()


if __name__ == "__main__":
    main()