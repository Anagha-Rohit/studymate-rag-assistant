"""Question-answering helpers for StudyMate.

This file connects retrieved note chunks to a chat model so answers come from
the student's uploaded notes.
"""

from src.config import require_openai_api_key

ANSWER_NOT_FOUND = "The notes do not contain enough information to answer this question."


PROMPT_TEMPLATE = """
You are StudyMate, a helpful revision assistant for university students.

Use only the retrieved notes below to answer the question.
Do not use outside knowledge.
If the notes do not contain the answer, say exactly:
"The notes do not contain enough information to answer this question."

Retrieved notes:
{context}

Question:
{question}

Answer:
""".strip()


def _create_chat_model():
    """Create the OpenAI chat model used for answering questions."""
    from langchain_openai import ChatOpenAI

    require_openai_api_key()

    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _get_page_content(document) -> str:
    """Get text from a LangChain Document in a beginner-friendly way."""
    return getattr(document, "page_content", "")


def _get_metadata(document) -> dict:
    """Get metadata from a LangChain Document."""
    return getattr(document, "metadata", {})


def _format_documents_for_prompt(documents) -> str:
    """Turn retrieved documents into text that can go inside the prompt."""
    prompt_parts = []

    for index, document in enumerate(documents, start=1):
        chunk_text = _get_page_content(document)
        prompt_parts.append(f"Source chunk {index}:\n{chunk_text}")

    return "\n\n".join(prompt_parts)


def _format_source_chunks(documents) -> list[dict]:
    """Return source chunks in a simple format for the Streamlit app."""
    source_chunks = []

    for index, document in enumerate(documents, start=1):
        metadata = _get_metadata(document)
        chunk_number = metadata.get("chunk_number", index)

        source_chunks.append(
            {
                "chunk_number": chunk_number,
                "text": _get_page_content(document),
                "metadata": metadata,
            }
        )

    return source_chunks


def answer_question(vector_store, question: str) -> dict:
    """Answer a question using only chunks retrieved from the vector store."""
    if not question.strip():
        return {"answer": "Please enter a question.", "source_chunks": []}

    # Ask the vector store for the chunks that best match the question.
    try:
        retrieved_documents = vector_store.similarity_search(question, k=4)
    except Exception as error:
        raise ValueError(
            "Could not search your uploaded notes. Try uploading the file again."
        ) from error

    if not retrieved_documents:
        return {"answer": ANSWER_NOT_FOUND, "source_chunks": []}

    context = _format_documents_for_prompt(retrieved_documents)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    try:
        chat_model = _create_chat_model()
        response = chat_model.invoke(
            [
                (
                    "system",
                    "You answer questions using only the provided study notes.",
                ),
                ("human", prompt),
            ]
        )
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(
            "Could not get an answer from OpenAI. Check your API key, billing, "
            "or internet connection, then try again."
        ) from error

    return {
        "answer": response.content,
        "source_chunks": _format_source_chunks(retrieved_documents),
    }
