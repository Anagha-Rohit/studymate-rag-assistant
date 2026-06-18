"""Shared configuration helpers for StudyMate."""

import os


def require_openai_api_key() -> str:
    """Return the OpenAI API key or raise a beginner-friendly error."""
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "OPENAI_API_KEY is missing. Create a .env file and add your real "
            "OpenAI API key before using AI features."
        )

    return api_key
