# StudyMate

StudyMate is a beginner-friendly Python project for a first-year university student.
It is planned as a Streamlit study assistant that uses LangChain, ChromaDB, and
retrieval-augmented generation to answer questions from uploaded lecture notes.

The project is intentionally built in small steps so each part of the RAG
workflow is easy to understand.

## Project Overview

StudyMate will let a student upload notes, split the notes into chunks, store
those chunks in a vector database, and ask questions about the material. The app
should answer from the uploaded notes and show the source chunks used, so the
student can check where the answer came from.

Later, StudyMate will also help students create flashcards and practice quiz
questions from their notes.

## Learning Goals

- Learn how a Streamlit app is organized.
- Learn how text can be loaded from PDF and TXT files.
- Learn why long notes need to be split into chunks.
- Learn what embeddings are and how they help with search.
- Learn how ChromaDB stores and retrieves note chunks.
- Learn how LangChain can connect retrieval with a language model.
- Learn how to show sources so answers can be checked.
- Learn how to keep API keys safe with environment variables.

## Planned Features

- Upload lecture notes as PDF or TXT files
- Split notes into smaller chunks
- Create embeddings from the chunks
- Store chunks in ChromaDB
- Ask questions about the uploaded notes
- Answer using retrieved notes instead of made-up information
- Show the source chunks used for each answer
- Generate flashcards
- Generate quiz questions

## Project Structure

- `app.py` - Main Streamlit app
- `src/ingest.py` - Loads notes, splits text, and stores chunks
- `src/rag_chain.py` - Retrieves notes and answers questions
- `src/flashcards.py` - Creates flashcards from notes
- `src/quiz_generator.py` - Creates quiz questions from notes
- `sample_data/sample_notes.txt` - Small example notes for testing
- `tests/test_basic.py` - Basic tests for the starter project

## Getting Started

1. Create and activate a virtual environment.
2. Install the requirements:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your own API key later.
4. Run the starter app:

```bash
streamlit run app.py
```

This is only the starting structure. The real RAG features will be added in small steps.
