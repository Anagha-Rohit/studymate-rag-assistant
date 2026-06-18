"""Flashcard generation helpers for StudyMate.

This file creates question-and-answer flashcards from uploaded lecture notes.
"""

from src.config import require_openai_api_key

FLASHCARD_NOT_FOUND = "The notes do not contain enough information to make flashcards."


FLASHCARD_PROMPT_TEMPLATE = """
You are StudyMate, a helpful revision assistant for university students.

Create {number} flashcards using only the retrieved notes below.
Do not use outside knowledge.
If the notes do not contain enough information, say exactly:
"The notes do not contain enough information to make flashcards."

Topic:
{topic}

Retrieved notes:
{context}

Format each flashcard like this:
Question: ...
Answer: ...
""".strip()


def _create_chat_model():
    """Create the OpenAI chat model used for flashcard generation."""
    from langchain_openai import ChatOpenAI

    require_openai_api_key()

    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _get_page_content(document) -> str:
    """Get text from a LangChain Document."""
    return getattr(document, "page_content", "")


def _format_documents_for_prompt(documents) -> str:
    """Turn retrieved documents into text for the flashcard prompt."""
    prompt_parts = []

    for index, document in enumerate(documents, start=1):
        chunk_text = _get_page_content(document)
        prompt_parts.append(f"Source chunk {index}:\n{chunk_text}")

    return "\n\n".join(prompt_parts)


def generate_flashcards(vector_store, topic=None, number=5) -> str:
    """Generate flashcards using only chunks from the vector store."""
    topic_text = topic.strip() if topic else ""
    search_query = topic_text or "important ideas from these notes"
    number = max(1, int(number))

    # Retrieve note chunks first so the model only sees relevant course notes.
    try:
        retrieved_documents = vector_store.similarity_search(search_query, k=number)
    except Exception as error:
        raise ValueError(
            "Could not search your uploaded notes for flashcards. Try uploading "
            "the file again."
        ) from error

    if not retrieved_documents:
        return FLASHCARD_NOT_FOUND

    context = _format_documents_for_prompt(retrieved_documents)
    prompt = FLASHCARD_PROMPT_TEMPLATE.format(
        number=number,
        topic=topic_text or "General notes",
        context=context,
    )

    try:
        chat_model = _create_chat_model()
        response = chat_model.invoke(
            [
                (
                    "system",
                    "You create flashcards using only the provided study notes.",
                ),
                ("human", prompt),
            ]
        )
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(
            "Could not generate flashcards with OpenAI. Check your API key, "
            "billing, or internet connection, then try again."
        ) from error

    return response.content
