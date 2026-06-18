"""Basic tests for the StudyMate starter project."""

from pathlib import Path

from src.flashcards import generate_flashcards
from src.ingest import split_text
from src.quiz_generator import generate_quiz_questions


def test_project_files_exist():
    """Check that the main starter files are present."""
    expected_files = [
        "app.py",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "README.md",
        "AGENTS.md",
        "src/ingest.py",
        "src/rag_chain.py",
        "src/quiz_generator.py",
        "src/flashcards.py",
        "sample_data/sample_notes.txt",
    ]

    for file_path in expected_files:
        assert Path(file_path).exists()


def test_split_text_creates_chunks():
    """Check that long text is split into smaller pieces."""
    chunks = split_text("abcdef", chunk_size=2)

    assert chunks == ["ab", "cd", "ef"]


def test_study_helpers_return_lists():
    """Check that starter quiz and flashcard helpers return lists."""
    assert isinstance(generate_quiz_questions("Photosynthesis notes"), list)
    assert isinstance(generate_flashcards("Photosynthesis notes"), list)
