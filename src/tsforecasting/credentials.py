"""
This module is used to manage credentials and sensitive information.
It is intended to be used in conjunction with a .env file to load
environment variables without hardcoding them in the codebase.
"""

from dotenv import load_dotenv

load_dotenv()

# Build credentials from environment variables
# e.g: MY_CREDENTIAL = os.getenv("MY_CREDENTIAL")
