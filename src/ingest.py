"""Helpers for loading notes and preparing them for retrieval.

Later this file will load PDF/TXT files, split text into chunks, create
embeddings, and save everything in ChromaDB.
"""


def load_text_file(file_path: str) -> str:
    """Read a plain text file and return its contents."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_txt_file(uploaded_file) -> str:
    """Read a TXT file uploaded with Streamlit and return its text.

    Streamlit gives uploaded files as bytes. We try to decode those bytes as
    UTF-8. If the file has a few unusual characters, we replace them instead of
    crashing the app.
    """
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
    from pypdf import PdfReader

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
    except Exception as error:
        raise ValueError("Could not read this PDF file.") from error

    text = "\n\n".join(page_texts).strip()

    if not text:
        raise ValueError("No readable text was found in this PDF.")

    return text


def split_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into simple chunks.

    This starter version splits by character count. Later we can replace it
    with a LangChain text splitter.
    """
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]
