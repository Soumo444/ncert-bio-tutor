---
title: NCERT Bio Tutor
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# NCERT Bio Tutor

A ChatGPT-style study assistant for the 14 NCERT Class 12 Biology books.

- Answers from the books with file + page citations when relevant
- Falls back to labelled general knowledge for other questions
- Practice questions and answer grading, chapter by chapter
- Groq `openai/gpt-oss-120b` for answers, Gemini for embeddings + fallback

The index is pre-built and this public demo is read-only: visitors can chat,
but cannot upload, reindex or delete documents.

Built with FastAPI, ChromaDB and a custom HTML/CSS/JS interface.
