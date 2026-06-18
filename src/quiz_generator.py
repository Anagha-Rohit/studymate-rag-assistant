"""Quiz generation helpers for StudyMate.

This file creates practice quiz questions from uploaded lecture notes.
"""

import json


QUIZ_NOT_FOUND = "The notes do not contain enough information to make a quiz."
EXAM_MODE_NOT_FOUND = "The notes do not contain enough information for exam mode."


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


EXAM_MODE_PROMPT_TEMPLATE = """
You are StudyMate, a helpful revision assistant for university students.

Create {number} exam-style questions using only the retrieved notes below.
Do not use outside knowledge.
The questions should test understanding, application, comparison, or explanation.
Avoid questions that only ask for simple definitions.
Include a mark scheme or model answer for every question.

Topic:
{topic}

Retrieved notes:
{context}

Return only valid JSON in this format:
[
  {{
    "question": "Question text",
    "model_answer": "Mark scheme or model answer"
  }}
]
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


def _clean_json_text(text: str) -> str:
    """Remove simple Markdown code fences from model JSON output."""
    clean_text = text.strip()

    if clean_text.startswith("```json"):
        clean_text = clean_text.removeprefix("```json").strip()
    elif clean_text.startswith("```"):
        clean_text = clean_text.removeprefix("```").strip()

    if clean_text.endswith("```"):
        clean_text = clean_text.removesuffix("```").strip()

    return clean_text


def _parse_exam_mode_response(text: str) -> list[dict[str, str]]:
    """Turn model JSON into a simple list of exam questions."""
    clean_text = _clean_json_text(text)
    raw_questions = json.loads(clean_text)
    exam_questions = []

    for item in raw_questions:
        question = str(item.get("question", "")).strip()
        model_answer = str(item.get("model_answer", "")).strip()

        if question and model_answer:
            exam_questions.append(
                {
                    "question": question,
                    "model_answer": model_answer,
                }
            )

    return exam_questions


def generate_exam_mode_questions(vector_store, topic=None, number=10) -> list[dict[str, str]]:
    """Generate exam-style questions with model answers from the notes."""
    topic_text = topic.strip() if topic else ""
    search_query = topic_text or "exam-style understanding questions from these notes"
    number = min(10, max(3, int(number)))

    # Retrieve note chunks first so the exam questions stay grounded in the notes.
    retrieved_documents = vector_store.similarity_search(search_query, k=number)

    if not retrieved_documents:
        return []

    context = _format_documents_for_prompt(retrieved_documents)
    prompt = EXAM_MODE_PROMPT_TEMPLATE.format(
        number=number,
        topic=topic_text or "General notes",
        context=context,
    )

    chat_model = _create_chat_model()
    response = chat_model.invoke(
        [
            (
                "system",
                "You create exam-style questions using only the provided study notes.",
            ),
            ("human", prompt),
        ]
    )

    try:
        return _parse_exam_mode_response(response.content)
    except (json.JSONDecodeError, TypeError, AttributeError):
        return [
            {
                "question": "Review the generated exam practice below.",
                "model_answer": response.content,
            }
        ]


def generate_quiz_questions(text: str) -> list[str]:
    """Return simple starter quiz questions."""
    if not text.strip():
        return []

    return [
        "What is the main idea of these notes?",
        "What are two important details from these notes?",
    ]
