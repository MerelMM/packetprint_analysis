# This script was developed with assistance from ChatGPT (OpenAI) and Github Copilot
# Final implementation and adaptation by Merel Haenaets.
def get_app_key(session_name: str) -> str:
    """
    Extract the app name from a folder name formatted like:
    session_appname_timestamp
    If it cannot parse, returns the full folder name.
    """
    parts = session_name.split("_")
    if len(parts) >= 3:
        return parts[1]  # 'appname' is typically the second field
    return session_name
