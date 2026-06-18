"""Quiz generation helpers for StudyMate.

Later this file will create practice quiz questions from retrieved notes.
"""


def generate_quiz_questions(text: str) -> list[str]:
    """Return simple starter quiz questions."""
    if not text.strip():
        return []

    return [
        "What is the main idea of these notes?",
        "What are two important details from these notes?",
    ]
