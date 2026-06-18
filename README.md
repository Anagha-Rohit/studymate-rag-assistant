# StudyMate

StudyMate is a beginner-friendly Python project for a first-year university student.

The goal is to build a LangChain + Streamlit RAG app that helps students study from their own lecture notes.

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
