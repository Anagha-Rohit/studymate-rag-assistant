"""Quiz generation helpers for StudyMate.

This file creates practice quiz questions from uploaded lecture notes.
"""

QUIZ_NOT_FOUND = "The notes do not contain enough information to make a quiz."


QUIZ_PROMPT_TEMPLATE = """
You are StudyMate, a helpful revision assistant for university students.

Create {number} quiz questions using only the retrieved notes below.
Do not use outside knowledge.
Include a mix of multiple-choice and short-answer questions.
Put all answers at the bottom under the heading "Answers".
If the notes do not contain enough information, say exactly:
"The notes do not contain enough information to make a quiz."

Topic:
{topic}

Retrieved notes:
{context}

Use this simple format:
1. [Multiple choice] Question...
   A. ...
   B. ...
   C. ...
   D. ...

2. [Short answer] Question...

Answers:
1. ...
2. ...
""".strip()


def _create_chat_model():
    """Create the OpenAI chat model used for quiz generation."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _get_page_content(document) -> str:
    """Get text from a LangChain Document."""
    return getattr(document, "page_content", "")


def _format_documents_for_prompt(documents) -> str:
    """Turn retrieved documents into text for the quiz prompt."""
    prompt_parts = []

    for index, document in enumerate(documents, start=1):
        chunk_text = _get_page_content(document)
        prompt_parts.append(f"Source chunk {index}:\n{chunk_text}")

    return "\n\n".join(prompt_parts)


def generate_quiz(vector_store, topic=None, number=5) -> str:
    """Generate a quiz using only chunks from the vector store."""
    topic_text = topic.strip() if topic else ""
    search_query = topic_text or "important ideas from these notes"
    number = max(1, int(number))

    # Retrieve note chunks first so quiz questions are grounded in the notes.
    retrieved_documents = vector_store.similarity_search(search_query, k=number)

    if not retrieved_documents:
        return QUIZ_NOT_FOUND

    context = _format_documents_for_prompt(retrieved_documents)
    prompt = QUIZ_PROMPT_TEMPLATE.format(
        number=number,
        topic=topic_text or "General notes",
        context=context,
    )

    chat_model = _create_chat_model()
    response = chat_model.invoke(
        [
            (
                "system",
                "You create quizzes using only the provided study notes.",
            ),
            ("human", prompt),
        ]
    )

    return response.content


def generate_quiz_questions(text: str) -> list[str]:
    """Return simple starter quiz questions."""
    if not text.strip():
        return []

    return [
        "What is the main idea of these notes?",
        "What are two important details from these notes?",
    ]
