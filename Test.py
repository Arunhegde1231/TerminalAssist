"""Quick test script to debug command execution"""

import subprocess

def test_subprocess_directly():
    """Test subprocess execution directly to see what we should expect"""
    print("=== Testing subprocess directly ===")
    try:
        result = subprocess.run(['ls', '-la'], capture_output=True, text=True, timeout=10)
        print(f"Return code: {result.returncode}")
        print(f"Stdout length: {len(result.stdout)}")
        print(f"Stderr length: {len(result.stderr)}")
        print("\n--- STDOUT ---")
        print(repr(result.stdout))
        print("\n--- STDERR ---")
        print(repr(result.stderr))
    except Exception as e:
        print(f"Error: {e}")

def test_shell_command():
    """Test shell command execution"""
    print("\n=== Testing shell command ===")
    try:
        result = subprocess.run('ls -la', shell=True, capture_output=True, text=True, timeout=10)
        print(f"Return code: {result.returncode}")
        print(f"Stdout length: {len(result.stdout)}")
        print(f"Stderr length: {len(result.stderr)}")
        print("\n--- STDOUT ---")
        print(repr(result.stdout))
        print("\n--- STDERR ---")
        print(repr(result.stderr))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_subprocess_directly()
    test_shell_command()