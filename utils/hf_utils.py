"""
Utilities for Hugging Face Hub authentication and operations.
"""

import os
from huggingface_hub import login


def authenticate_huggingface(token: str = None):
    """
    Authenticate with Hugging Face Hub to access gated models.
    
    Args:
        token: Hugging Face token. If None, will check environment variables
               HUGGINGFACE_TOKEN or HF_TOKEN, or prompt for interactive login.
    
    Example:
        >>> authenticate_huggingface()  # Uses env var or prompts
        >>> authenticate_huggingface(token="hf_...")  # Uses provided token
    """
    if token is None:
        token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    
    if token:
        login(token=token)
        print("✅ Successfully authenticated with Hugging Face Hub")
    else:
        print("⚠️  No token provided. Attempting interactive login...")
        print("   You can also set HUGGINGFACE_TOKEN or HF_TOKEN environment variable")
        login()  # Interactive login

