"""Basic tests for the StudyMate starter project."""

from pathlib import Path

from src.flashcards import FLASHCARD_NOT_FOUND, generate_flashcards
from src.ingest import (
    create_vector_store,
    load_pdf_file,
    load_txt_file,
    split_text,
    split_text_into_chunks,
)
from src.quiz_generator import (
    EXAM_MODE_NOT_FOUND,
    QUIZ_NOT_FOUND,
    generate_exam_mode_questions,
    generate_quiz,
    generate_quiz_questions,
)
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


class FixedSizeTextSplitter:
    """Small test helper that splits text without needing LangChain installed."""

    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    def split_text(self, text: str) -> list[str]:
        return [
            text[index : index + self.chunk_size]
            for index in range(0, len(text), self.chunk_size)
        ]


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


def test_split_text_into_chunks_returns_list(monkeypatch):
    """Check that chunking returns a normal Python list."""
    monkeypatch.setattr(
        "src.ingest._create_text_splitter",
        lambda: FixedSizeTextSplitter(chunk_size=20),
    )

    chunks = split_text_into_chunks("These are short notes.")

    assert isinstance(chunks, list)


def test_split_text_into_chunks_creates_multiple_chunks_for_long_text(monkeypatch):
    """Check that long notes are split into more than one chunk."""
    monkeypatch.setattr(
        "src.ingest._create_text_splitter",
        lambda: FixedSizeTextSplitter(chunk_size=50),
    )

    long_text = "StudyMate helps students revise from uploaded notes. " * 10
    chunks = split_text_into_chunks(long_text)

    assert len(chunks) > 1


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


def test_generate_flashcards_uses_retrieved_notes(monkeypatch):
    """Check that flashcards are generated from retrieved source chunks."""
    saved_inputs = {}

    class FakeVectorStore:
        def similarity_search(self, question, k):
            saved_inputs["question"] = question
            saved_inputs["k"] = k

            return [
                FakeDocument("A variable stores a value.", {"chunk_number": 1}),
                FakeDocument("A loop repeats code.", {"chunk_number": 2}),
            ]

    class FakeResponse:
        content = "Question: What does a variable do?\nAnswer: It stores a value."

    class FakeChatModel:
        def invoke(self, messages):
            saved_inputs["messages"] = messages
            return FakeResponse()

    monkeypatch.setattr("src.flashcards._create_chat_model", lambda: FakeChatModel())

    flashcards = generate_flashcards(FakeVectorStore(), topic="variables", number=3)

    assert "Question: What does a variable do?" in flashcards
    assert "Answer: It stores a value." in flashcards
    assert saved_inputs["question"] == "variables"
    assert saved_inputs["k"] == 3
    assert "Do not use outside knowledge" in saved_inputs["messages"][1][1]


def test_generate_flashcards_uses_general_notes_without_topic(monkeypatch):
    """Check that flashcards can be generated without a topic."""
    saved_inputs = {}

    class FakeVectorStore:
        def similarity_search(self, question, k):
            saved_inputs["question"] = question
            return [FakeDocument("Algorithms solve problems.", {"chunk_number": 1})]

    class FakeResponse:
        content = "Question: What do algorithms do?\nAnswer: They solve problems."

    class FakeChatModel:
        def invoke(self, messages):
            return FakeResponse()

    monkeypatch.setattr("src.flashcards._create_chat_model", lambda: FakeChatModel())

    flashcards = generate_flashcards(FakeVectorStore())

    assert "Question:" in flashcards
    assert saved_inputs["question"] == "important ideas from these notes"


def test_generate_flashcards_handles_no_retrieved_notes():
    """Check that StudyMate gives a clear message if no notes are retrieved."""

    class FakeVectorStore:
        def similarity_search(self, question, k):
            return []

    flashcards = generate_flashcards(FakeVectorStore(), topic="biology")

    assert flashcards == FLASHCARD_NOT_FOUND


def test_generate_quiz_uses_retrieved_notes(monkeypatch):
    """Check that quiz questions are generated from retrieved source chunks."""
    saved_inputs = {}

    class FakeVectorStore:
        def similarity_search(self, question, k):
            saved_inputs["question"] = question
            saved_inputs["k"] = k

            return [
                FakeDocument("A function is a reusable block of code.", {"chunk_number": 1}),
                FakeDocument("A parameter is an input to a function.", {"chunk_number": 2}),
            ]

    class FakeResponse:
        content = (
            "1. [Multiple choice] What is a function?\n"
            "   A. A reusable block of code\n"
            "   B. A database\n\n"
            "2. [Short answer] What is a parameter?\n\n"
            "Answers:\n"
            "1. A\n"
            "2. An input to a function."
        )

    class FakeChatModel:
        def invoke(self, messages):
            saved_inputs["messages"] = messages
            return FakeResponse()

    monkeypatch.setattr("src.quiz_generator._create_chat_model", lambda: FakeChatModel())

    quiz = generate_quiz(FakeVectorStore(), topic="functions", number=4)

    assert "[Multiple choice]" in quiz
    assert "[Short answer]" in quiz
    assert "Answers:" in quiz
    assert saved_inputs["question"] == "functions"
    assert saved_inputs["k"] == 4
    assert "Do not use outside knowledge" in saved_inputs["messages"][1][1]


def test_generate_quiz_uses_general_notes_without_topic(monkeypatch):
    """Check that a quiz can be generated without a topic."""
    saved_inputs = {}

    class FakeVectorStore:
        def similarity_search(self, question, k):
            saved_inputs["question"] = question
            return [FakeDocument("Data can be stored in lists.", {"chunk_number": 1})]

    class FakeResponse:
        content = "1. [Short answer] What can store data?\n\nAnswers:\n1. Lists."

    class FakeChatModel:
        def invoke(self, messages):
            return FakeResponse()

    monkeypatch.setattr("src.quiz_generator._create_chat_model", lambda: FakeChatModel())

    quiz = generate_quiz(FakeVectorStore())

    assert "Answers:" in quiz
    assert saved_inputs["question"] == "important ideas from these notes"


def test_generate_quiz_handles_no_retrieved_notes():
    """Check that StudyMate gives a clear message if no quiz notes are found."""

    class FakeVectorStore:
        def similarity_search(self, question, k):
            return []

    quiz = generate_quiz(FakeVectorStore(), topic="chemistry")

    assert quiz == QUIZ_NOT_FOUND


def test_generate_exam_mode_questions_uses_retrieved_notes(monkeypatch):
    """Check that Exam Mode creates questions and model answers from notes."""
    saved_inputs = {}

    class FakeVectorStore:
        def similarity_search(self, question, k):
            saved_inputs["question"] = question
            saved_inputs["k"] = k

            return [
                FakeDocument("Algorithms solve problems step by step.", {"chunk_number": 1}),
                FakeDocument("Debugging means finding and fixing errors.", {"chunk_number": 2}),
            ]

    class FakeResponse:
        content = """
        [
          {
            "question": "Explain how an algorithm can help solve a problem.",
            "model_answer": "A good answer explains that an algorithm gives clear steps."
          },
          {
            "question": "Why is debugging useful when writing programs?",
            "model_answer": "Debugging helps find and fix errors in code."
          }
        ]
        """

    class FakeChatModel:
        def invoke(self, messages):
            saved_inputs["messages"] = messages
            return FakeResponse()

    monkeypatch.setattr("src.quiz_generator._create_chat_model", lambda: FakeChatModel())

    questions = generate_exam_mode_questions(FakeVectorStore(), topic="algorithms", number=6)

    assert len(questions) == 2
    assert questions[0]["question"] == "Explain how an algorithm can help solve a problem."
    assert "clear steps" in questions[0]["model_answer"]
    assert saved_inputs["question"] == "algorithms"
    assert saved_inputs["k"] == 6
    assert "Avoid questions that only ask for simple definitions" in saved_inputs["messages"][1][1]


def test_generate_exam_mode_questions_uses_general_notes_without_topic(monkeypatch):
    """Check that Exam Mode works without a topic."""
    saved_inputs = {}

    class FakeVectorStore:
        def similarity_search(self, question, k):
            saved_inputs["question"] = question
            saved_inputs["k"] = k
            return [FakeDocument("Loops repeat instructions.", {"chunk_number": 1})]

    class FakeResponse:
        content = """
        [
          {
            "question": "Explain why loops are useful in a program.",
            "model_answer": "Loops are useful because they repeat instructions."
          }
        ]
        """

    class FakeChatModel:
        def invoke(self, messages):
            return FakeResponse()

    monkeypatch.setattr("src.quiz_generator._create_chat_model", lambda: FakeChatModel())

    questions = generate_exam_mode_questions(FakeVectorStore(), number=20)

    assert questions[0]["question"].startswith("Explain why loops")
    assert saved_inputs["question"] == "exam-style understanding questions from these notes"
    assert saved_inputs["k"] == 10


def test_generate_exam_mode_questions_handles_no_retrieved_notes():
    """Check that Exam Mode returns no questions if no notes are retrieved."""

    class FakeVectorStore:
        def similarity_search(self, question, k):
            return []

    questions = generate_exam_mode_questions(FakeVectorStore(), topic="databases")

    assert questions == []
    assert EXAM_MODE_NOT_FOUND == "The notes do not contain enough information for exam mode."


def test_study_helpers_return_lists():
    """Check that starter quiz helpers return lists."""
    assert isinstance(generate_quiz_questions("Photosynthesis notes"), list)
