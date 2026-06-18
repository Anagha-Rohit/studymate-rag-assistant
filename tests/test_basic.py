"""Basic tests for the StudyMate starter project."""

from pathlib import Path

from src.flashcards import generate_flashcards
from src.ingest import load_pdf_file, load_txt_file, split_text, split_text_into_chunks
from src.quiz_generator import generate_quiz_questions


class FakeUploadedFile:
    """Small test helper that acts like a Streamlit uploaded file."""

    def __init__(self, file_bytes: bytes):
        self._file_bytes = file_bytes

    def getvalue(self) -> bytes:
        return self._file_bytes


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


def test_load_txt_file_reads_uploaded_text():
    """Check that uploaded TXT bytes become a normal Python string."""
    uploaded_file = FakeUploadedFile("Hello, StudyMate!".encode("utf-8"))

    assert load_txt_file(uploaded_file) == "Hello, StudyMate!"


def test_load_txt_file_handles_decoding_errors():
    """Check that unusual bytes do not crash the TXT loader."""
    uploaded_file = FakeUploadedFile(b"Hello\xffStudyMate")

    text = load_txt_file(uploaded_file)

    assert "Hello" in text
    assert "StudyMate" in text


def test_load_pdf_file_raises_error_when_no_text(monkeypatch):
    """Check that a PDF with no readable text gives a simple error."""

    class FakePage:
        def extract_text(self):
            return None

    class FakeReader:
        pages = [FakePage()]

    monkeypatch.setattr("src.ingest._create_pdf_reader", lambda uploaded_file: FakeReader())

    uploaded_file = FakeUploadedFile(b"%PDF test bytes")

    try:
        load_pdf_file(uploaded_file)
    except ValueError as error:
        assert str(error) == "No readable text was found in this PDF."
    else:
        raise AssertionError("Expected load_pdf_file to raise ValueError.")


def test_split_text_into_chunks_uses_text_splitter(monkeypatch):
    """Check that StudyMate uses the configured text splitter."""

    class FakeTextSplitter:
        def split_text(self, text):
            return [text[:5], text[5:]]

    monkeypatch.setattr(
        "src.ingest._create_text_splitter",
        lambda: FakeTextSplitter(),
    )

    chunks = split_text_into_chunks("StudyMate notes")

    assert chunks == ["Study", "Mate notes"]


def test_split_text_into_chunks_handles_empty_text():
    """Check that empty notes return no chunks."""
    assert split_text_into_chunks("   ") == []


def test_study_helpers_return_lists():
    """Check that starter quiz and flashcard helpers return lists."""
    assert isinstance(generate_quiz_questions("Photosynthesis notes"), list)
    assert isinstance(generate_flashcards("Photosynthesis notes"), list)
