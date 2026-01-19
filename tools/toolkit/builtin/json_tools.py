from tools.decorator import tool
import json

@tool()
def json_is_valid(s: str) -> bool:
    """
    Check if the input string is valid JSON.

    Args:
        s: The string to check.

    Returns:
        True if the string is valid JSON, False otherwise.
    """
    try:
        json.loads(s)
        return True
    except ValueError:
        return False

if __name__ == "__main__":
    print(json_is_valid.to_string())