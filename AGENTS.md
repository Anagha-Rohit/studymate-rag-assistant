# AGENTS.md

This file gives guidance to coding assistants working on StudyMate.

## Project Goal

StudyMate should stay simple enough for a first-year university student to understand.

## Coding Style

- Use clear Python names.
- Prefer small functions with short comments.
- Avoid clever code when simple code works.
- Do not add real API keys or secrets.
- Keep each new feature easy to test.

## Planned Modules

- `app.py` is for the Streamlit user interface.
- `src/ingest.py` is for loading and chunking notes.
- `src/rag_chain.py` is for retrieval and question answering.
- `src/flashcards.py` is for making flashcards.
- `src/quiz_generator.py` is for making quiz questions.
