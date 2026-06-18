# StudyMate Streamlit entry point.
# This file will become the main web app that students run with Streamlit.

import streamlit as st


st.set_page_config(page_title="StudyMate")

st.title("StudyMate")
st.write("A beginner-friendly RAG study assistant for lecture notes.")

st.info(
    "Starter project only: uploading notes, retrieval, flashcards, and quizzes "
    "will be added step by step."
)

uploaded_file = st.file_uploader("Upload lecture notes", type=["pdf", "txt"])
question = st.text_input("Ask a question about your notes")

if uploaded_file:
    st.success("File uploaded. Ingestion will be connected later.")

if question:
    st.write("Question received. The RAG chain will answer this later.")
