# Terminal Assist

Terminal Assist is an AI-powered, user-friendly application designed to transform and enhance your command-line interface (CLI) interactions. Running directly within your terminal, it aims to make your daily tasks more efficient, intuitive, and less prone to errors.

## How to Run

To get Terminal Assist up and running on your system, follow these steps:

1.  **Install Python:**

    *   **Windows:** Download and install Python from the official website: [https://www.python.org/downloads/](https://www.python.org/downloads/)
    *   **macOS:** Open your terminal and run the following command using Homebrew:
        ```bash
        brew install python
        ```

2.  **Clone the Repository:**

    Open your terminal and clone the Terminal Assist repository from GitHub:

    ```bash
    git clone https://github.com/Arunhegde1231/TerminalAssist.git
    ```

3.  **Create a Virtual Environment:**

    Navigate into the `TerminalAssist` directory in your terminal and create a virtual environment to manage dependencies:

    *   **Windows:**
        ```bash
        python -m venv .venv
        ```
    *   **macOS/Linux:**
        ```bash
        python3 -m venv .venv
        ```

4.  **Activate the Virtual Environment:**

    Activate the newly created virtual environment to isolate your project dependencies:

    *   **Windows:**
        ```bash
        .venv\Scripts\activate.bat
        ```
    *   **macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```

5.  **Install Dependencies:**

    With your virtual environment activated, install all necessary project dependencies using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

6.  **Run the Application:**

    Finally, launch Terminal Assist from within the `TerminalAssist` directory:

    *   **Windows:**
        ```bash
        python HomePage.py
        ```
    *   **macOS/Linux:**
        ```bash
        python3 HomePage.py
        ```
