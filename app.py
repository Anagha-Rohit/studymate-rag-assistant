# StudyMate Streamlit entry point.
# This file is the first simple version of the web app.

import streamlit as st

from src.ingest import load_pdf_file, load_txt_file, split_text_into_chunks


st.set_page_config(page_title="StudyMate")

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

# This will later be sent to the RAG system.
question = st.text_area("Ask a question about your notes")

# These buttons are placeholders for features we will build next.
ask_clicked = st.button("Ask question")
flashcards_clicked = st.button("Generate flashcards")
quiz_clicked = st.button("Generate quiz")

if uploaded_file is not None:
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

        if note_text:
            chunks = split_text_into_chunks(note_text)

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
                with st.expander("Show first chunk"):
                    st.write(chunks[0])
    except ValueError as error:
        st.error(str(error))

if ask_clicked:
    st.info("AI question answering will be added next.")

if flashcards_clicked:
    st.info("AI flashcard generation will be added next.")

if quiz_clicked:
    st.info("AI quiz generation will be added next.")

st.write(
    "Friendly note: the AI features are not connected yet. "
    "They will be added in the next steps."
)
