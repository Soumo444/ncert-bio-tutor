# NCERT Bio Tutor AI

A free AI study assistant for **NCERT Class 12 Biology**. Ask a question in
plain English and get an answer built from the 14 NCERT PDFs — with the chapter
and page cited.

## 🚀 Live App

### 👉 [**ncert-bio-tutor.onrender.com**](https://ncert-bio-tutor.onrender.com)

**No login. No install. Free to use on phone or desktop.**

[Source code](https://github.com/Soumo444/ncert-bio-tutor) ·
[Built with RAG, Groq and Gemini](#tech-stack)

---

## What is this project?

This is a **Generative AI study assistant** built on top of the NCERT Class 12
Biology textbooks. You ask a question, and instead of guessing, the app first
searches the actual book pages and then asks an AI model to answer from them.

That approach is called **RAG — Retrieval-Augmented Generation**. In simple
words:

> **Search the books first, then let the AI explain.**

That is why the answer can show a citation like `lebo104.pdf · page 8` instead
of inventing facts. If the books do not cover the question, the app can answer
from general knowledge and clearly labels it.

The project also supports **practice questions**, **chapter-wise quizzes**, and
**grading of your written answers**.

---

## Features

| Feature | What it does |
|---|---|
| **Answers from the books** | Searches 706 indexed chunks from 14 NCERT PDFs and cites chapter + page |
| **Two answer styles** | **ChatGPT** (books first, general knowledge fallback) or **Docs only** (strictly from the books) |
| **Practice questions** | “Give me a practice question from chapter 4” |
| **Quiz grading** | Write your answer, and the bot grades it and explains what you missed |
| **Chapter-aware search** | Understands “chapter 4” or the chapter's topic and stays inside that chapter |
| **General knowledge fallback** | Answers outside the books, clearly labelled — no fake citations |
| **Streaming answers** | Text appears as it is generated, with sources shown first |
| **Read-only public demo** | Visitors can chat, but cannot upload, reindex or delete anything |

---

## Try asking

| Question | What happens |
|---|---|
| `What is a test cross?` | Answers from the PDFs with citations |
| `Give me a practice question from chapter 4.` | Creates an exam-style question from that chapter |
| `Explain the lac operon.` | Answers from the Molecular Basis of Inheritance chapter |
| `Who won the 2026 FIFA World Cup final?` | General answer, labelled `general knowledge` |
| `hello` | Friendly reply, no search and no model call |

---

## How it works

```text
Your question
     │
     ▼
Search the 14 NCERT PDFs (706 chunks)
     │
     ▼
Pick the 6 most useful passages
     │
     ▼
Groq writes the answer (Gemini as backup)
     │
     ▼
Answer + chapter/page citations
```

1. **Indexing (one time):** each PDF is split into small overlapping chunks.
   Every chunk is converted into a vector (an embedding) and stored in ChromaDB.
2. **Retrieval:** your question is converted into a vector too, and the closest
   chunks are selected. A re-ranking step keeps the results relevant and avoids
   near-duplicate passages.
3. **Generation:** Groq's `gpt-oss-120b` writes the answer using those chunks.
   If Groq is rate-limited, the app automatically falls back to Gemini.
4. **Verification:** in Docs-only mode, a second AI call checks whether the
   answer is actually supported by the passages.

---

## Tech stack

| Layer | Technology |
|---|---|
| **Frontend** | Custom HTML, CSS and JavaScript (no framework) |
| **Backend** | FastAPI (Python 3.12) |
| **Vector database** | ChromaDB |
| **Embeddings** | Google Gemini (`gemini-embedding-001`) |
| **Answer model** | Groq `openai/gpt-oss-120b` |
| **Fallback model** | Gemini 3.1 Flash Lite → Gemini 3.5 Flash |
| **Hosting** | Render free web service (Docker) |
| **Deployment bundle** | Public Hugging Face dataset |

---

## Run it locally

### 1. Requirements

- Windows (the helper scripts are `.bat` files)
- Python 3.12
- Free **Groq** and **Gemini** API keys

### 2. Install

```bat
git clone https://github.com/Soumo444/ncert-bio-tutor.git
cd ncert-bio-tutor

uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
copy .env.example .env
```

### 3. Add your free API keys

| Key | Purpose | Get it from |
|---|---|---|
| `GROQ_API_KEY` | Writes the answers | <https://console.groq.com/keys> |
| `GEMINI_API_KEY` | Creates embeddings + runs the answer check | <https://aistudio.google.com/apikey> |

Put them in `.env`:

```env
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AQ...
```

Both are free and need no credit card. The keys stay on the server and are
never sent to the browser.

### 4. Add the PDFs

The repository does **not** include the PDFs (they are ~160 MB). Download the
NCERT Class 12 Biology books and place them here:

```text
data\docs\lebo1dd\lebo101.pdf
data\docs\lebo1dd\lebo102.pdf
...
```

Subfolders are allowed. Supported formats: **PDF, DOCX, TXT, MD, CSV, XLSX**.

### 5. Build the index

```bat
.venv\Scripts\python.exe ingest.py
```

For the 14 NCERT books this creates **706 chunks**. Unchanged files are skipped
on later runs.

| Command | Effect |
|---|---|
| `ingest.py --force` | Re-embed everything |
| `ingest.py --reset` | Wipe the index first |
| `ingest.py --file lebo104.pdf` | Index one file |
| `ingest.py --chunk-size 700 --overlap 120` | Try different chunking |

### 6. Start the app

```bat
start.bat
```

Or run it directly:

```bat
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open **<http://127.0.0.1:8000>**.

---

## Deployment

### Live app

**<https://ncert-bio-tutor.onrender.com>**

### How it is hosted

The public version runs on **Render's free plan**:

- No credit card
- 512 MB RAM (the app uses about 90 MB)
- Permanent `https://...onrender.com` address
- Sleeps after **15 minutes** without visitors; the next visit takes about
  **1 minute** to wake up

The public deployment is **read-only**:

- Uploads, reindexing and deletes return **HTTP 403**
- Each visitor is limited to **60 questions per hour**
- The PDFs themselves are not downloadable

### Deployment architecture

```text
GitHub repo (Dockerfile only)
        │
        ▼
Render builds a Docker image
        │
        ▼
Docker downloads the 4.8 MB app bundle
(app + 706-chunk index, no PDFs, no keys)
        │
        ▼
FastAPI starts with Uvicorn
        │
        ▼
https://ncert-bio-tutor.onrender.com
```

The bundle is stored in a public Hugging Face dataset:

<https://huggingface.co/datasets/Soumojit444/ncert-bio-tutor-deploy>

### Updating the live app

1. Build a fresh bundle:

```bat
.venv\Scripts\python.exe deploy\build_bundle.py
```

2. Upload the new `deploy\bundle.tar.gz` to the Hugging Face dataset
   (replace the existing file).

3. In Render, choose **Manual Deploy → Clear build cache & deploy**.

> **Important:** clearing the cache is required. Otherwise Render reuses the
> old download layer and keeps serving the previous version.

The Docker build repository is separate and contains only the Dockerfile and
pinned requirements:

<https://github.com/Soumo444/ncert-bio-tutor-render>

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | App status, models, chunk and document counts |
| `POST` | `/api/chat` | Stream an answer (SSE) with citations |
| `GET` | `/api/documents` | List indexed documents |
| `POST` | `/api/upload` | Upload a document (blocked in the public demo) |
| `POST` | `/api/reindex` | Rebuild the index (blocked in the public demo) |
| `DELETE` | `/api/documents/{path}` | Delete a document (blocked in the public demo) |

Chat is a **streaming** response. Each event is a JSON line beginning with
`data:`, ending with a `done` event that contains the citations and the model
used.

---

## Evaluation

The app includes an evaluation script so retrieval quality is measured, not
guessed.

```bat
:: Retrieval only - instant, no LLM calls
.venv\Scripts\python.exe eval\run_eval.py --retrieval-only

:: Full run with answers and the AI judge
.venv\Scripts\python.exe eval\run_eval.py --pause 35
```

### Results for the 14 NCERT PDFs (706 chunks)

| Metric | Score |
|---|---:|
| hit@6 (right chapter in the top 6) | **1.00** |
| MRR (rank of the first correct chunk) | **0.974** |
| precision@6 | **0.78** |
| context recall | **1.00** |
| faithfulness | **1.00** |
| answer relevance | **1.00** |
| keyword recall | **1.00** |
| abstention accuracy | **1.00** |

The correct chapter ranked **#1 for 18 of 19** answerable questions and **#2**
for the other. Precision is below 1 because chapters naturally overlap, not
because retrieval picks wrong documents.

---

## Project structure

```text
app/
  main.py              FastAPI routes: chat, upload, documents, health
  templates/index.html Chat page
  static/style.css     Design
  static/app.js        Streaming, citations, sources panel

rag/
  config.py            Settings loaded from .env
  loaders.py           PDF, DOCX, TXT/MD, CSV and XLSX readers
  chunking.py          Splits documents into overlapping chunks
  embeddings.py        Gemini embeddings with batching and retries
  store.py             ChromaDB storage
  retriever.py         Vector search + MMR re-ranking
  llm.py               Groq, Gemini, OpenRouter and Ollama fallback
  prompts.py           Answer and judge prompts
  pipeline.py          Retrieve → answer → cite → verify

eval/                  Evaluation scripts and question set
ingest.py              Command-line indexer
start.bat              One-click local start
share.bat              Temporary public link (optional)

deploy/
  build_bundle.py      Builds the app + index bundle for Render
  render_deploy.py     Creates the Render service
  render_repo/         Dockerfile and pinned requirements
  github_upload.py     Syncs the Dockerfile files to GitHub
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GEMINI_API_KEY is missing` | Add the key to `.env` and restart |
| Groq answers are slow or use Gemini | Groq's free daily quota is finished; the app switched to the slower backup |
| `429` from Groq | Free per-minute limit; the app waits and retries automatically |
| PDF returns no text | The PDF is scanned images; OCR is not included |
| “Couldn't find that” | Check the document is indexed (`/api/documents`) or lower `MIN_SCORE` |
| First visit is slow | Render was asleep; free services wake in about 1 minute |
| Port 8000 already in use | Change `PORT` in `.env` |

---

## Free-tier notes

- **Groq** answers roughly **80–100 simple questions per day** across the whole
  public app. Long chats and Strong mode use more.
- When the Groq daily limit is reached, the app **automatically switches to
  Gemini**. The site keeps working, but answers take longer.
- **Gemini** is also used for embeddings and the groundedness check.
- Render's free service sleeps after 15 minutes idle. A free uptime monitor
  (for example UptimeRobot checking `/api/health` every 5 minutes) keeps it warm.

Quotas change over time. The app is designed to fall back instead of breaking.

---

## Notes and limitations

- The index covers the **14 NCERT Class 12 Biology PDFs** supplied for this
  project. Class 11 topics (for example osmosis) are answered as general
  knowledge, without fake citations.
- Scanned image-only PDFs need OCR before indexing.
- The public demo is intentionally read-only and rate-limited.
- This is a personal educational project, not an official NCERT product.

---

## Credits

- **NCERT** — Class 12 Biology textbooks and chapter content.
- **Groq** — free `gpt-oss-120b` inference.
- **Google Gemini** — free embeddings and fallback generation.
- **FastAPI**, **ChromaDB**, **Render** and **Hugging Face** — the open-source
  and free-tier tools that make the project run.
