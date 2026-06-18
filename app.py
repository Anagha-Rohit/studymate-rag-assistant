# StudyMate Streamlit entry point.
# This file is the first simple version of the web app.

from dotenv import load_dotenv
import streamlit as st

from src.flashcards import generate_flashcards
from src.ingest import (
    create_vector_store,
    load_pdf_file,
    load_txt_file,
    split_text_into_chunks,
)
from src.quiz_generator import (
    EXAM_MODE_NOT_FOUND,
    generate_exam_mode_questions,
    generate_quiz,
)
from src.rag_chain import answer_question


load_dotenv()

st.set_page_config(page_title="StudyMate")


def show_pipeline_status(file_uploaded, text_extracted, chunks_created, vector_store_ready):
    """Show the current progress through the StudyMate RAG pipeline."""
    st.subheader("Pipeline status")

    steps = [
        ("1. File uploaded", file_uploaded),
        ("2. Text extracted", text_extracted),
        ("3. Chunks created", chunks_created),
        ("4. Vector store ready", vector_store_ready),
    ]

    for label, is_done in steps:
        status = "Done" if is_done else "Waiting"
        st.write(f"{label}: {status}")


st.sidebar.title("How to use StudyMate")
st.sidebar.write("1. Upload a TXT or PDF file with your lecture notes.")
st.sidebar.write("2. Wait for StudyMate to extract and prepare the text.")
st.sidebar.write("3. Ask a question about the uploaded notes.")
st.sidebar.write("4. Check the source chunks used for the answer.")
st.sidebar.info("Your API key stays in your `.env` file and is not shown in the app.")

st.title("StudyMate: AI Revision Assistant")

st.write(
    "StudyMate will help you revise from your own lecture notes. "
    "You will be able to upload PDF or TXT files, ask questions, and later "
    "generate flashcards and quizzes from your notes."
)

# Students can upload their lecture notes here.
uploaded_file = st.file_uploader(
    "Upload your lecture notes",
    type=["pdf", "txt"],
)

file_uploaded = uploaded_file is not None
text_extracted = False
chunks_created = False
vector_store_ready = False

# These inputs are used by the RAG, flashcard, quiz, and Exam Mode features.
question = st.text_area("Ask a question about your notes")
flashcard_topic = st.text_input("Optional flashcard topic")
quiz_topic = st.text_input("Optional quiz topic")

st.subheader("Exam Mode")
exam_topic = st.text_input("Optional exam topic")
exam_question_count = st.slider(
    "Number of exam questions",
    min_value=3,
    max_value=10,
    value=10,
)

# These buttons run the main StudyMate study features.
ask_clicked = st.button("Ask question")
flashcards_clicked = st.button("Generate flashcards")
quiz_clicked = st.button("Generate quiz")
exam_mode_clicked = st.button("Generate exam mode questions")

if uploaded_file is None:
    st.info("No file uploaded yet. Upload a TXT or PDF file to begin.")
else:
    file_name = uploaded_file.name
    file_type = file_name.split(".")[-1].upper()
    note_text = ""

    try:
        if file_name.lower().endswith(".txt"):
            note_text = load_txt_file(uploaded_file)
        elif file_name.lower().endswith(".pdf"):
            note_text = load_pdf_file(uploaded_file)
        else:
            st.warning("Please upload a TXT or PDF file.")

        if not note_text.strip():
            st.session_state.pop("vector_store", None)
            st.session_state.pop("uploaded_file_key", None)
            st.error("This file has no readable text. Try another TXT or PDF file.")
        else:
            text_extracted = True
            chunks = split_text_into_chunks(note_text)
            chunks_created = len(chunks) > 0

            st.success("File loaded successfully.")
            st.write(f"File name: {file_name}")
            st.write(f"Detected file type: {file_type}")
            st.write(f"Number of characters extracted: {len(note_text)}")
            st.write(f"Number of chunks created: {len(chunks)}")

            # Show only the beginning so long lecture notes do not fill the page.
            st.text_area(
                "Preview of your notes",
                value=note_text[:1000],
                height=250,
                disabled=True,
            )

            if chunks:
                file_key = f"{file_name}-{len(note_text)}"

                if st.session_state.get("uploaded_file_key") != file_key:
                    st.session_state.pop("vector_store", None)
                    st.session_state["vector_store"] = create_vector_store(chunks)
                    st.session_state["uploaded_file_key"] = file_key

                vector_store_ready = st.session_state.get("uploaded_file_key") == file_key
                st.success("Vector store is ready for questions.")

                with st.expander("Show first chunk"):
                    st.write(chunks[0])
            else:
                st.session_state.pop("vector_store", None)
                st.session_state.pop("uploaded_file_key", None)
                st.error("No chunks were created from this file. Try a file with more text.")
    except ValueError as error:
        st.session_state.pop("vector_store", None)
        st.session_state.pop("uploaded_file_key", None)
        st.error(str(error))
    except Exception:
        st.session_state.pop("vector_store", None)
        st.session_state.pop("uploaded_file_key", None)
        st.error(
            "Could not create the vector store. Check that OPENAI_API_KEY is set "
            "in your .env file."
        )

show_pipeline_status(file_uploaded, text_extracted, chunks_created, vector_store_ready)

if ask_clicked:
    if uploaded_file is None:
        st.error("Please upload notes before asking a question.")
    elif "vector_store" not in st.session_state:
        st.error("Your notes are not ready yet. Check the upload and vector store status.")
    elif not question.strip():
        st.error("Please type a question before clicking Ask question.")
    else:
        try:
            result = answer_question(st.session_state["vector_store"], question)

            st.subheader("Answer")
            st.write(result["answer"])

            source_chunks = result["source_chunks"]

            if source_chunks:
                st.subheader("Source chunks")

                for index, source_chunk in enumerate(source_chunks, start=1):
                    chunk_number = source_chunk["chunk_number"]

                    with st.expander(f"Source {index}: chunk {chunk_number}"):
                        st.write(source_chunk["text"])
        except Exception:
            st.error(
                "Could not answer the question. Check that OPENAI_API_KEY is set "
                "in your .env file."
            )

if flashcards_clicked:
    if uploaded_file is None:
        st.error("Please upload notes before generating flashcards.")
    elif "vector_store" not in st.session_state:
        st.error("Your notes are not ready yet. Check the upload and vector store status.")
    else:
        try:
            flashcards = generate_flashcards(
                st.session_state["vector_store"],
                topic=flashcard_topic,
                number=5,
            )

            st.subheader("Flashcards")

            if flashcard_topic.strip():
                st.write(f"Topic: {flashcard_topic}")
            else:
                st.write("Topic: General notes")

            st.text(flashcards)
        except Exception:
            st.error(
                "Could not generate flashcards. Check that OPENAI_API_KEY is set "
                "in your .env file."
            )

if quiz_clicked:
    if uploaded_file is None:
        st.error("Please upload notes before generating a quiz.")
    elif "vector_store" not in st.session_state:
        st.error("Your notes are not ready yet. Check the upload and vector store status.")
    else:
        try:
            quiz = generate_quiz(
                st.session_state["vector_store"],
                topic=quiz_topic,
                number=5,
            )

            st.subheader("Quiz")

            if quiz_topic.strip():
                st.write(f"Topic: {quiz_topic}")
            else:
                st.write("Topic: General notes")

            st.text(quiz)
        except Exception:
            st.error(
                "Could not generate the quiz. Check that OPENAI_API_KEY is set "
                "in your .env file."
            )

if exam_mode_clicked:
    if uploaded_file is None:
        st.error("Please upload notes before using Exam Mode.")
    elif "vector_store" not in st.session_state:
        st.error("Your notes are not ready yet. Check the upload and vector store status.")
    else:
        try:
            exam_questions = generate_exam_mode_questions(
                st.session_state["vector_store"],
                topic=exam_topic,
                number=exam_question_count,
            )

            st.subheader("Exam Mode Questions")

            if exam_topic.strip():
                st.write(f"Topic: {exam_topic}")
            else:
                st.write("Topic: General notes")

            if not exam_questions:
                st.warning(EXAM_MODE_NOT_FOUND)
            else:
                st.write("Questions")

                for index, exam_question in enumerate(exam_questions, start=1):
                    st.write(f"{index}. {exam_question['question']}")

                st.write("Model answers")

                for index, exam_question in enumerate(exam_questions, start=1):
                    with st.expander(f"Model answer {index}"):
                        st.write(exam_question["model_answer"])
        except Exception:
            st.error(
                "Could not generate Exam Mode questions. Check that OPENAI_API_KEY "
                "is set in your .env file."
            )

st.write(
    "Friendly note: StudyMate uses the notes you upload for answers, flashcards, "
    "quizzes, and Exam Mode."
)
