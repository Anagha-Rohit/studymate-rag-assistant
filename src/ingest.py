"""Helpers for loading notes and preparing them for retrieval."""

from src.config import require_openai_api_key


def load_txt_file(uploaded_file) -> str:
    """Read a TXT file uploaded with Streamlit and return its text.

    Streamlit gives uploaded files as bytes. We try to decode those bytes as
    UTF-8. If the file has a few unusual characters, we replace them instead of
    crashing the app.
    """
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    file_bytes = uploaded_file.getvalue()

    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("utf-8", errors="replace")


def _create_pdf_reader(uploaded_file):
    """Create a PDF reader.

    The import stays inside this helper so the rest of the project can still be
    tested even before students install all optional packages.
    """
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise ValueError(
            "The PDF package is missing. Run `pip install -r requirements.txt`, "
            "then restart Streamlit."
        ) from error

    return PdfReader(uploaded_file)


def load_pdf_file(uploaded_file) -> str:
    """Extract text from a PDF file uploaded with Streamlit.

    PDFs store text page by page. We use pypdf to read each page, collect the
    readable text, and join everything into one string for StudyMate.
    """
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    try:
        reader = _create_pdf_reader(uploaded_file)
        page_texts = []

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                page_texts.append(page_text)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("Could not read this PDF file.") from error

    text = "\n\n".join(page_texts).strip()

    if not text:
        raise ValueError("No readable text was found in this PDF.")

    return text


def _create_text_splitter():
    """Create the LangChain text splitter used for RAG chunks."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )


def split_text_into_chunks(text: str) -> list[str]:
    """Split a document into smaller chunks for RAG.

    RAG apps do not usually send a whole textbook or lecture file to the AI at
    once. Instead, we split the text into smaller overlapping chunks so the app
    can search for the most useful pieces later.
    """
    if not text.strip():
        return []

    text_splitter = _create_text_splitter()
    chunks = text_splitter.split_text(text)

    return [chunk for chunk in chunks if chunk.strip()]


def _create_openai_embeddings():
    """Create the OpenAI embedding model used by LangChain."""
    require_openai_api_key()

    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as error:
        raise ValueError(
            "The LangChain OpenAI package is missing. Run "
            "`pip install -r requirements.txt`, then restart Streamlit."
        ) from error

    return OpenAIEmbeddings()


def _create_chroma_vector_store(chunks: list[str], metadata: list[dict], embeddings):
    """Create a Chroma vector store from text chunks and metadata."""
    try:
        from langchain_chroma import Chroma
    except ImportError as error:
        raise ValueError(
            "The Chroma package is missing. Run `pip install -r requirements.txt`, "
            "then restart Streamlit."
        ) from error

    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadata,
        collection_name="studymate_notes",
    )


def create_vector_store(chunks: list[str]):
    """Create a Chroma vector store from text chunks.

    Embeddings are number-based summaries of text. They help the computer find
    chunks that are similar in meaning, even if the exact words are different.
    ChromaDB stores those embeddings so StudyMate can search the notes later.
    """
    clean_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    if not clean_chunks:
        raise ValueError("No text chunks were provided for the vector store.")

    metadata = []

    for index, _chunk in enumerate(clean_chunks, start=1):
        metadata.append({"chunk_number": index})

    try:
        embeddings = _create_openai_embeddings()
        return _create_chroma_vector_store(clean_chunks, metadata, embeddings)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(
            "Could not create the vector store. Check your OpenAI API key and "
            "try uploading the notes again."
        ) from error
