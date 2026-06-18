# StudyMate Streamlit entry point.
# This file is the first simple version of the web app.

import streamlit as st

from src.ingest import load_txt_file


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
    if uploaded_file.name.lower().endswith(".txt"):
        note_text = load_txt_file(uploaded_file)

        st.success("TXT file loaded successfully.")
        st.write(f"File name: {uploaded_file.name}")
        st.write(f"Number of characters: {len(note_text)}")

        # Show only the beginning so long lecture notes do not fill the page.
        st.text_area(
            "Preview of your notes",
            value=note_text[:1000],
            height=250,
            disabled=True,
        )
    else:
        st.info("PDF support will be added later. For now, please try a TXT file.")

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
