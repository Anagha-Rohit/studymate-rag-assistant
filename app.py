# StudyMate Streamlit entry point.
# This file is the first simple version of the web app.

import streamlit as st


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
    st.success("Your file has been uploaded. Note processing will be added next.")

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
