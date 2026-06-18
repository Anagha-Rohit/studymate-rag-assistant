"""Basic tests for the StudyMate starter project."""

from pathlib import Path

from src.flashcards import generate_flashcards
from src.ingest import (
    create_vector_store,
    load_pdf_file,
    load_txt_file,
    split_text,
    split_text_into_chunks,
)
from src.quiz_generator import generate_quiz_questions
from src.rag_chain import ANSWER_NOT_FOUND, answer_question


class FakeUploadedFile:
    """Small test helper that acts like a Streamlit uploaded file."""

    def __init__(self, file_bytes: bytes):
        self._file_bytes = file_bytes

    def getvalue(self) -> bytes:
        return self._file_bytes


class FakeDocument:
    """Small test helper that acts like a LangChain Document."""

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


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


def test_create_vector_store_adds_chunk_metadata(monkeypatch):
    """Check that chunks are stored with simple chunk number metadata."""
    saved_inputs = {}

    class FakeEmbeddings:
        pass

    class FakeVectorStore:
        pass

    def fake_create_vector_store(chunks, metadata, embeddings):
        saved_inputs["chunks"] = chunks
        saved_inputs["metadata"] = metadata
        saved_inputs["embeddings"] = embeddings
        return FakeVectorStore()

    monkeypatch.setattr("src.ingest._create_openai_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr("src.ingest._create_chroma_vector_store", fake_create_vector_store)

    vector_store = create_vector_store([" First chunk ", "Second chunk"])

    assert isinstance(vector_store, FakeVectorStore)
    assert saved_inputs["chunks"] == ["First chunk", "Second chunk"]
    assert saved_inputs["metadata"] == [{"chunk_number": 1}, {"chunk_number": 2}]
    assert isinstance(saved_inputs["embeddings"], FakeEmbeddings)


def test_create_vector_store_rejects_empty_chunks():
    """Check that empty chunks do not create an empty vector store."""
    try:
        create_vector_store([" ", ""])
    except ValueError as error:
        assert str(error) == "No text chunks were provided for the vector store."
    else:
        raise AssertionError("Expected create_vector_store to raise ValueError.")


def test_answer_question_uses_retrieved_notes(monkeypatch):
    """Check that questions are answered from retrieved source chunks."""
    saved_inputs = {}

    class FakeVectorStore:
        def similarity_search(self, question, k):
            saved_inputs["question"] = question
            saved_inputs["k"] = k

            return [
                FakeDocument("Photosynthesis uses sunlight.", {"chunk_number": 1}),
                FakeDocument("Plants make glucose.", {"chunk_number": 2}),
            ]

    class FakeResponse:
        content = "Photosynthesis uses sunlight to help plants make glucose."

    class FakeChatModel:
        def invoke(self, messages):
            saved_inputs["messages"] = messages
            return FakeResponse()

    monkeypatch.setattr("src.rag_chain._create_chat_model", lambda: FakeChatModel())

    result = answer_question(FakeVectorStore(), "What does photosynthesis use?")

    assert result["answer"] == "Photosynthesis uses sunlight to help plants make glucose."
    assert result["source_chunks"][0]["chunk_number"] == 1
    assert result["source_chunks"][0]["text"] == "Photosynthesis uses sunlight."
    assert saved_inputs["question"] == "What does photosynthesis use?"
    assert saved_inputs["k"] == 4
    assert "Do not use outside knowledge" in saved_inputs["messages"][1][1]


def test_answer_question_handles_no_retrieved_notes():
    """Check that StudyMate gives a clear answer when no notes match."""

    class FakeVectorStore:
        def similarity_search(self, question, k):
            return []

    result = answer_question(FakeVectorStore(), "What is mitosis?")

    assert result == {"answer": ANSWER_NOT_FOUND, "source_chunks": []}


def test_study_helpers_return_lists():
    """Check that starter quiz and flashcard helpers return lists."""
    assert isinstance(generate_quiz_questions("Photosynthesis notes"), list)
    assert isinstance(generate_flashcards("Photosynthesis notes"), list)
