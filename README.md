# AI study assistant over your documents

Ask anything. The assistant searches your PDFs, Word files, text notes and
spreadsheets first, cites the file and page when it uses them, and falls back to
general knowledge when the documents do not cover the question — clearly
labelled, like ChatGPT. A strict **Docs only** switch restores the original
"answer only from the files" behaviour.

**Cloud, free, no model downloads.** Answers are written by a 120B open model on
Groq's free tier; embeddings and the groundedness judge run on Google's free tier.
Nothing heavy is installed on your PC.

---

## 1. The models (measured on this machine, Nov 2026)

| Mode | Model | What actually happens |
|---|---|---|
| **Fast** (default) | `gpt-oss-120b` on Groq, `reasoning_effort=low` | ~1 s answers, strong 120B open model |
| **Strong** (one click) | same model, `reasoning_effort=medium` + 3 000-token budget | thinks longer, better on multi-step questions |
| Automatic fallback | Gemini 3.1 Flash Lite → Gemini 3.5 Flash | used when Groq is rate-limited |

Measured facts about the free tier (they matter):

* **Gemini Pro is not free.** On a new key the quota is literally `limit: 0` — requests
  return 429 forever, no matter how long you wait.
* **Gemini 3.8 Flash was returning 503** (overloaded) during setup; **3.5 Flash works
  but took 30–80 s per answer**, which is unusable for chat.
* **Groq `gpt-oss-120b` answers in ~1 s**, has a 131k context window and a free tier.
  That is why it is the primary brain.
* The **Gemini key is still essential**: it powers the embeddings
  (`gemini-embedding-001`), which is fast and has a free quota.

### Living within the free tiers

* **Embeddings:** free tier counts every text as a request, 100/minute. Ingestion uses
  batches of 30 with a 21 s pause, so ~700 chunks index in about 6 minutes without 429s.
* **Groq answers:** 8 000 tokens/minute. When the window is exhausted the app waits for
  the reset and retries (usually a few seconds) instead of silently switching to a slow model.
* **Groundedness judge:** runs on Gemini, so it uses a separate quota and does not steal
  tokens from the answering model.

So "strong, free, fast" is satisfied by the pair, not by one vendor: Groq for
generation, Gemini for embeddings (+ backup generation). If Groq's daily limit is
hit, the app silently switches to Gemini and keeps working.

Anyone who opens the web page can chat — they do **not** need a key or any setup.

---

## Answer styles

Pick a style in **Settings → Answer style**. The default is **ChatGPT**.

| Style | Behaviour |
|---|---|
| **ChatGPT** (default) | Documents first. If the files cover the question, the answer cites file + page. Otherwise it answers normally and the UI shows a `general knowledge` badge. Unused PDF snippets are hidden. |
| **Docs only** | Strict RAG. The assistant refuses when the files do not contain the answer. The groundedness judge runs only in this mode. |

Things you can type:

| Message | What happens |
|---|---|
| `hello` | Friendly reply, no retrieval and no model call |
| `Give me a practice question from chapter 4.` | One exam-style question + answer from that chapter |
| `What is a test cross?` | Answer with citations from the PDFs |
| `Mendel crossed...` after a practice question | The bot grades your attempt and explains |
| `Who won the 2026 FIFA World Cup final?` | General answer, labelled, no fake citations |

Chapter requests understand both `chapter 4` and the chapter's topic, and
`one more question` stays in the same chapter. The local chapter map covers the
13 NCERT Biology chapters (`lebo101.pdf` … `lebo113.pdf`).

---

## 2. Install (once)

Requirements: Windows, Python 3.12 (the `start.bat` script installs it via `uv`).

```bat
cd "F:\AI  CHATBOT RAG"
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
copy .env.example .env
```

### Add your two free API keys

Both are free, no credit card:

1. **Groq** writes the answers — <https://console.groq.com/keys> → *Create API key*.
2. **Gemini** makes the embeddings and runs the groundedness judge —
   <https://aistudio.google.com/apikey>.

Paste them into `.env`:

```
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AQ...
```

The app needs both: Gemini turns your documents into vectors, Groq writes the
answer. Neither key is ever sent to the browser.

Optional third provider: <https://openrouter.ai/keys> → `OPENROUTER_API_KEY=`.

---

## 3. Add documents

Put files in `data\docs\` (subfolders allowed):

```
data\docs\biology\chapter1.pdf
data\docs\notes.md
data\docs\prices.xlsx
```

Supported: **PDF, DOCX, TXT, MD, CSV, XLSX**. You can also drag-and-drop in the UI.

## 4. Index them

```bat
.venv\Scripts\python.exe ingest.py
```

```
> lebo101.pdf (35.1 MB) parsing...
> lebo101.pdf: 25 page(s) -> 69 chunks, embedding...
  + lebo101.pdf: stored 69 chunks
```

Unchanged files are skipped on the next run. Flags:

| Command | Effect |
|---|---|
| `ingest.py --force` | re-embed everything |
| `ingest.py --reset` | wipe the index first |
| `ingest.py --file x.pdf` | ingest one file |
| `ingest.py --chunk-size 700 --overlap 120` | experiment with chunking |

## 5. Run

```bat
start.bat
```

or

```bat
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>.

To let others on your Wi-Fi use it: change `HOST=0.0.0.0` in `.env` and give them
`http://YOUR-PC-IP:8000` (allow port 8000 through Windows Firewall).

To share it over the internet, double-click **`share.bat`** (or run it from a
terminal). It starts the server if needed, creates the free tunnel, and prints a
public link:

```bat
share.bat
```

Or do it manually while the server is running:

```bat
cloudflared tunnel --url http://127.0.0.1:8000
```

The first run downloads `cloudflared.exe` once. Cloudflare prints a public
`https://...trycloudflare.com` link you can send to anyone. The link changes each
time you start the tunnel. Anyone with the link can chat (no key or install
needed) and can also see your documents, so only share links you are comfortable
with, and stop the tunnel when you are done sharing.

---

## 5b. Deploy free on Render (live)

**Live app: <https://ncert-bio-tutor.onrender.com>**

Full application source (public): <https://github.com/Soumo444/ncert-bio-tutor>

The app runs on **Render's free web service**: no credit card, 512 MB RAM (the
container uses ~90 MB), and a permanent `https://...onrender.com` URL. Visitors
can chat; uploads, reindexing and deletes are blocked (`READ_ONLY_MODE=true`) and
each IP is limited to 60 questions per hour (`RATE_LIMIT_PER_HOUR=60`).

Hugging Face Docker Spaces were the original plan, but free `cpu-basic` Docker
Spaces now require a paid PRO subscription (`402 Payment Required`), so Render is
used instead.

### How the deployment is wired

The service builds from a tiny public GitHub repository that contains only the
Dockerfile and pinned requirements:

```
https://github.com/Soumo444/ncert-bio-tutor-render
```

During the Docker build, the Dockerfile downloads the 4.8 MB application bundle
(app + 706-chunk Chroma index, no PDFs, no `.env`) from the public dataset:

```
https://huggingface.co/datasets/Soumojit444/ncert-bio-tutor-deploy
```

The tarball is used instead of a plain clone because Hugging Face LFS-tracks
`data/chroma/chroma.sqlite3` and would otherwise deliver pointer files.

### Re-deploying after a code change

1. Build a fresh bundle (creates `deploy/bundle.tar.gz`, no PDFs or `.env`):

```bat
.venv\Scripts\python.exe deploy\build_bundle.py
```

2. Upload the new `bundle.tar.gz` to the `ncert-bio-tutor-deploy` dataset
   (replace the existing file).

3. Trigger a manual deploy: Render Dashboard → **Manual Deploy → Clear build
   cache & deploy**. The cache must be cleared because Render otherwise reuses
   the cached layer that downloads `bundle.tar.gz` and keeps the old code.

The Dockerfile downloads the tarball again during the build. No application
files are copied into the Docker build repository.

### Free-plan behaviour

* The service **sleeps after 15 minutes without traffic** and wakes on the next
  request (~1 minute cold start). The URL never changes.
* **750 instance hours/month** are included, which is enough for one always-on
  service. A free uptime pinger (e.g. UptimeRobot, 5-minute HTTP check on
  `/api/health`) keeps it warm.
* The filesystem is ephemeral: everything is rebuilt from the image, which is
  why the index is baked into the bundle rather than uploaded at runtime.

### After deploying

The Render API key and the GitHub/Hugging Face tokens are only needed to create
or update the deployment. Revoke them once you are done:

* Render: <https://dashboard.render.com/u/settings?add-api-key>
* GitHub: <https://github.com/settings/tokens>
* Hugging Face: <https://huggingface.co/settings/tokens>

The live app keeps running without any of them.

---

## 6. How it works

```
PDF/DOCX/CSV ──parse──> pages ──chunk──> 900-char chunks
                                              │
                                     embed (Gemini)
                                              │
                                        ChromaDB (local)
                                              │
question ──embed──> vector search (top 24) ──> MMR rerank ──> 6 chunks
                                                              │
                                  ChatGPT mode: docs first, general fallback
                                  Docs-only mode: grounded prompt + citations
                                                              │
                                          Groq gpt-oss-120b (fallback: Gemini)
                                                              │
                                 groundedness judge (Docs-only mode)
```

### Chunking — `rag/chunking.py`
- Splits on paragraph → line → sentence → word boundaries, so sentences stay whole.
- 900 characters per chunk, 150 characters overlap (an answer on a boundary appears fully in one chunk).
- Each chunk gets a **context header** (`file | page 12 | section: Refunds`) that is embedded
  with the text and shown in the prompt.

### Retrieval — `rag/retriever.py`
- Cosine similarity in Chroma, `MIN_SCORE=0.20` drops nothing but the weakest chunks.
- **MMR** (Maximal Marginal Relevance) re-ranks 24 candidates into 6 that are relevant
  *and* different from each other — avoids six near-duplicate paragraphs.
- **Chapter filter:** a named chapter (`chapter 4`) restricts retrieval to that file, so
  practice questions really come from the chapter the user asked for.
- If nothing clears the threshold, **Docs only** refuses; **ChatGPT** mode answers from
  general knowledge instead of pretending the files matched.

### Hallucination mitigation — `rag/prompts.py`, `rag/pipeline.py`
1. **Docs only** allows only the numbered context blocks; **ChatGPT** tells the model to
   ignore irrelevant context and never invent a citation.
2. Every document fact must carry a citation `[2]`; the UI resolves it to file + page.
3. **Docs only** uses a fixed refusal string when the context lacks the answer (tested in
   the eval set). **ChatGPT** labels general-knowledge answers and hides unused snippets.
4. Documents are treated as data, not instructions (prompt-injection defence).
5. In **Docs only**, a second **judge call** on Gemini (separate free quota) scores
   faithfulness and flags unsupported claims — shown as a green / amber / red badge.

### Evaluation — `eval/run_eval.py`
Build a question set, then measure instead of guessing:

| Metric | Meaning |
|---|---|
| `hit@k` | did the right document appear in the top-k? |
| `MRR` | how high did the first correct chunk rank? |
| `precision@k` | how much of the retrieved context was relevant? |
| `context recall` | how many expected documents were found? |
| `faithfulness` | are all answer claims supported by context? (LLM judge) |
| `answer relevance` | does the answer address the question? |
| `keyword recall` | did expected facts appear in the answer? |
| `abstention accuracy` | does it correctly refuse unanswerable questions? |

```bat
:: retrieval only - instant, no LLM calls
.venv\Scripts\python.exe eval\run_eval.py --retrieval-only

:: full run: answers + LLM judge, paced for the free tier
.venv\Scripts\python.exe eval\run_eval.py --pause 35

:: your own questions
.venv\Scripts\python.exe eval\generate_questions.py --count 20
```

`eval\questions.json` ships with 21 hand-written questions mapped to the chapters,
including two questions the documents cannot answer (abstention tests). Reports are
written to `eval\reports\report-*.md`.

### Measured results (your 14 PDFs, 706 chunks, Nov 2026)

| Metric | Score |
|---|---|
| hit@6 | 1.00 |
| MRR | 0.974 |
| precision@6 | 0.78 |
| context recall | 1.00 |
| faithfulness | 1.00 |
| answer relevance | 1.00 |
| keyword recall | 1.00 |
| abstention accuracy | 1.00 |

The correct chapter ranked **#1 for 18 of 19** answerable questions and **#2** for the
other. Precision is below 1 because chapters genuinely overlap (e.g. the biotechnology
chapter mentions apomixis), not because retrieval picks wrong documents.

Full report: `eval\reports\report-20261118-151450.md`.

---

## 7. Experiments that teach the concepts

Change one thing, re-run `ingest.py --force` + `eval\run_eval.py`, compare the report.

| Experiment | Change | What to watch |
|---|---|---|
| Chunk size | `CHUNK_SIZE=400` vs `1600` | small = precise but fragmented; large = context but noisy |
| Overlap | `CHUNK_OVERLAP=0` vs `300` | boundary questions fail without overlap |
| No MMR | `USE_MMR=false` | duplicate chunks crowd out the right one |
| Strict threshold | `MIN_SCORE=0.45` | fewer wrong sources, more refusals |
| Strict grounding | `TEMPERATURE=0` | more consistent, less paraphrase |
| Bigger context | `TOP_K=3` vs `10` | recall up, distraction up, latency up |

---

## 8. Project map

```
app/
  main.py              FastAPI routes: chat (SSE), upload, documents, reindex, health
  templates/index.html chat UI
  static/style.css     design
  static/app.js        streaming, citations, sources panel
rag/
  config.py            all settings from .env
  loaders.py           PDF (PDFium), DOCX, TXT/MD, CSV, XLSX
  chunking.py          structure-aware splitter + context headers
  embeddings.py        Gemini embeddings, batching, retry/backoff
  store.py             ChromaDB persistence, delete/reindex
  retriever.py         vector search + MMR + score cutoff
  llm.py               Gemini / Groq / OpenRouter / Ollama with fallback
  prompts.py           grounded system prompt, judge prompt
  pipeline.py          retrieve → answer → cite → verify
eval/
  generate_questions.py
  run_eval.py
  questions.json       your test set
ingest.py              CLI indexer
start.bat              one-click start
deploy/
  build_bundle.py      builds deploy/bundle.tar.gz (app + index) for Render
  render_deploy.py     creates the Render web service and waits for it to go live
  render_repo/         Dockerfile + pinned requirements uploaded to GitHub
  github_upload.py     syncs the three Dockerfile files to the build repository
```

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `GEMINI_API_KEY is missing` | paste the key in `.env` and restart (needed for embeddings) |
| `429` / `RESOURCE_EXHAUSTED` on Groq | free per-minute token cap; the app now waits for the reset and retries, then falls back |
| `429` on Gemini **Pro** | Pro has zero free quota — Fast/Strong use Groq instead |
| PDF returns no text | scanned/image PDF — needs OCR (not included) |
| Answers say "couldn't find that" | check the document is indexed (`/api/documents`), lower `MIN_SCORE` |
| Slow first answer | cold start + embedding round-trip; later answers are faster |
| Port already in use | change `PORT=8000` in `.env` |

---

## 10. Notes on free tiers

Free-tier quotas change; the app fails over instead of breaking. For a public deployment
with many users, put a key on a server, add auth/rate limiting, and expect to move to a
paid tier eventually — Gemini Flash costs cents per thousand questions at that scale.
