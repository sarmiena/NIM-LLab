import subprocess
from typing import List, Dict, Any

def run_command(command: List[str]) -> Dict[str, Any]:
    """Execute a shell command and return the result."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        return {
            'success': True,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'stdout': e.stdout,
            'stderr': e.stderr,
            'returncode': e.returncode
        }
    except FileNotFoundError:
        return {
            'success': False,
            'error': f"Command not found: {command[0]}"
        }
