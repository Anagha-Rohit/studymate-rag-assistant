# AGENTS.md

This file gives guidance to coding assistants working on StudyMate.

## Project Goal

StudyMate should stay simple enough for a first-year university student to understand.
It is a Streamlit app that uses LangChain for RAG logic and ChromaDB as the
vector store for lecture note chunks.

## Coding Style

- Use clear Python names.
- Keep the code beginner-friendly.
- Prefer simple, readable Python over clever advanced code.
- Keep functions small and focused on one job.
- Add helpful comments when working with LangChain, RAG, embeddings, and vector stores.
- Avoid clever code when simple code works.
- Use Streamlit for the user interface.
- Use LangChain for retrieval-augmented generation logic.
- Use ChromaDB for the vector store.
- Keep each new feature easy to test.

## Secrets and Environment Variables

- Never commit API keys, tokens, passwords, or other secrets.
- Use environment variables for secrets.
- Keep example values in `.env.example`, but never add real secret values.

## Tests and Changes

- Add or update tests when changing project logic.
- Run relevant tests before finishing when possible.
- Explain what changed after editing files.

## Planned Modules

- `app.py` is for the Streamlit user interface.
- `src/ingest.py` is for loading and chunking notes.
- `src/rag_chain.py` is for retrieval and question answering.
- `src/flashcards.py` is for making flashcards.
- `src/quiz_generator.py` is for making quiz questions.
