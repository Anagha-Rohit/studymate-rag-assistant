"""Question-answering helpers for StudyMate.

Later this file will connect a retriever, prompt, and language model so answers
come from the student's uploaded notes.
"""


def answer_question(question: str) -> str:
    """Return a placeholder answer until the real RAG chain is built."""
    return (
        "StudyMate has not connected the RAG chain yet. "
        f"Your question was: {question}"
    )
