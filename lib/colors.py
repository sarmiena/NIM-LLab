def green_text(message: str) -> str:
    return f"\033[92m{message}\033[0m"

def red_text(message: str) -> str:
    return f"\033[91m{message}\033[0m"

def yellow_text(message: str) -> str:
    return f"\033[93m{message}\033[0m"
