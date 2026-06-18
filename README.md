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

1. Create a virtual environment:

```bash
python3 -m venv .venv
```

2. Activate the virtual environment:

```bash
source .venv/bin/activate
```

3. Install the requirements:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file from the example file:

```bash
cp .env.example .env
```

5. Add your OpenAI API key to `.env`:

```bash
OPENAI_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your own key. Do not commit your `.env` file.

6. Run the app:

```bash
streamlit run app.py
```

## Example Questions

After uploading `sample_data/sample_notes.txt`, try prompts like:

- Explain binary search in simple terms
- Make me 5 flashcards about Big O notation
- Create an exam quiz about recursion
- What are the key differences between arrays and linked lists?

## Running Tests

StudyMate uses `pytest` for simple project tests.

Run the tests with:

```bash
python3 -m pytest -q
```

These tests do not need a real OpenAI API key.

This is only the starting structure. The real RAG features will be added in small steps.
