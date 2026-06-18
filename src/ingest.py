"""Helpers for loading notes and preparing them for retrieval.

Later this file will load PDF/TXT files, split text into chunks, create
embeddings, and save everything in ChromaDB.
"""


def load_text_file(file_path: str) -> str:
    """Read a plain text file and return its contents."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def split_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into simple chunks.

    This starter version splits by character count. Later we can replace it
    with a LangChain text splitter.
    """
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]
