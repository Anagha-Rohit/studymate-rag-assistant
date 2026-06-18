"""Flashcard generation helpers for StudyMate.

Later this file will create question-and-answer flashcards from lecture notes.
"""


def generate_flashcards(text: str) -> list[dict[str, str]]:
    """Return a simple starter flashcard."""
    if not text.strip():
        return []

    return [
        {
            "front": "What topic do these notes discuss?",
            "back": "Review the notes and write the main topic in your own words.",
        }
    ]
