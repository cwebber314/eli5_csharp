# Explain it to me like I'm 5 — C# edition

A small pipeline for learning Retrieval-Augmented Generation (RAG) over a C#
codebase, running **fully offline**.

- **Embeddings:** `bge-small-en-v1.5` (local, in `models/`)
- **Vector store:** Chroma (persisted to `chroma_db/`)
- **Framework:** LangChain
- **Generation:** Qwen2.5-Coder-7B GGUF via Ollama (default; `qwen3-coder` 30B also supported)

## Pipeline steps

1. **Ingest** (this repo, done) — load C# files, chunk them C#-aware, embed with
   bge-small, store in Chroma.
2. **Retrieve** (done) — query the store for relevant code chunks.
3. **Generate** (done) — feed retrieved chunks + question to Qwen for
   answers, via four front-ends (one-shot CLI, terminal chat, history-aware
   chat, Gradio web UI).

## Architecture

Two phases. **Indexing** runs once per repo to build the vector store;
**querying** runs on every question. The embedding model is shared by both, so
questions and code land in the same vector space.

```
INDEXING  (run once per repo — ingest.py)

   repos/*.cs ─► load ─► C#-aware chunk ─► embed (bge-small) ─► Chroma
   (source)                                                    (chroma_db/)


QUERYING  (per question — rag.py orchestrates)

   question
      │
      ▼  (chat.py / app.py only, if prior turns exist)
   history-aware rewrite ──────────┐
      │                            │
      ▼                            ▼
   embed query (bge-small) ─► Chroma similarity search ─► top-k code chunks
                                                               │
                                     build grounded prompt ◄───┘
                                     (system + context + question)
                                               │
                                               ▼  llm.py
                                     Qwen  (via Ollama)
                                               │
                                               ▼
                                     answer + cited source files
```

### Module map

| File | Role |
|------|------|
| `ingest.py` | Indexing: load → chunk → embed → store. Also exposes `build_embedder()`, reused everywhere so retrieval matches ingestion. |
| `retrieve.py` | Inspect retrieval alone (scores + sources), no LLM — the debugging lens for the vector search. |
| `rag.py` | Shared RAG core: `answer()` / `stream_answer()`, prompt templates, history-aware query rewriting. |
| `llm.py` | Model factory — builds the Ollama chat model and holds sampling config. |
| `generate.py` | Front-end 1 — one-shot CLI, no memory. |
| `chat.py` | Front-ends 2 & 3 — terminal REPL with session memory + history-aware retrieval. |
| `app.py` | Front-end 4 — Gradio web chat with a cited-source code viewer. |
| `Modelfile` | Registers a local GGUF into Ollama without re-downloading. |
| `models/` | Local weights: `bge-small` / `modernbert` embedders + Qwen GGUF(s). |
| `chroma_db/` | Persisted vector store (created by `ingest.py`). |
| `repos/` | The C# source you ingest. |

All three chat front-ends call `rag.py`, which calls `llm.py` for generation and
reuses `ingest.py`'s embedder + Chroma for retrieval — one core, four faces.

## Download offline models

Download the ingest models like this:
```sh
pip install hf
hf download BAAI/bge-small-en-v1.5 --local-dir ./models/bge-small-en-v1.5
hf download nomic-ai/modernbert-embed-base --local-dir ./models/modernbert-embed-base
```

These include the pytorch model and safetensors model. You can delete the pytorch model if you want to save disk space
for whatever reason - they are the same model in a different format. 


And the LLM models:
```sh
hf download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF --local-dir ./models/qwen3-coder-gguf --include "*Q4_K_M*"
hf download bartowski/Qwen2.5-Coder-7B-Instruct-GGUF --local-dir ./models/qwen2.5-coder-7b-gguf --include "*Q4_K_M.gguf"
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ollama setup

You can download the windows zip here if you don't want to run an installer [link](https://github.com/ollama/ollama/releases)

## Get a sample C# repo

```bash
git clone --depth 1 https://github.com/KeRNeLith/QuikGraph.git repos/quikgraph
```

## Ingest

```bash
python ingest.py --source repos/quikgraph/src --reset
```

### Choosing the embedding model

Two embedders are available; switch with the `EMBED_MODEL` env var (default
`bge-small`). Each gets its own Chroma collection, so you can build both and
A/B them without clobbering the other.

| `EMBED_MODEL` | Model | Dim | Context |
|---------------|-------|-----|---------|
| `bge-small` (default) | `bge-small-en-v1.5` | 384 | 512 tok |
| `modernbert` | `nomic-ai/modernbert-embed-base` | 768 | 8192 tok |

```powershell
# Build a second store with modernbert (the bge store is left untouched)
$env:EMBED_MODEL = "modernbert"
python ingest.py --source repos/quikgraph/src --reset

# Compare retrieval between the two on the same query
python retrieve.py "how do I export to graphviz" --k 5
$env:EMBED_MODEL = "bge-small"
python retrieve.py "how do I export to graphviz" --k 5
```

Set `EMBED_MODEL` consistently for ingest **and** querying — it selects both the
embedder and the matching collection.

## Sanity-check the store

```bash
python ingest.py --query "how do I export to graphviz"
```

## Retrieve

`retrieve.py` embeds your query with the same offline embedding model and prints
the nearest code chunks, each with a normalized relevance score (0–1), its source
file, and its chunk index — so you can see exactly *what* the vector search found
before any LLM is involved.

```bash
# Basic query (defaults to the 4 nearest chunks)
python retrieve.py "how do I export to graphviz"

# Ask for more chunks
python retrieve.py "how do I export to graphviz" --k 8
```

Arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `query` (positional) | — | The text to search for. Quote it if it has spaces. |
| `--k` | 4 | Number of chunks to retrieve. |

Things to try to build intuition:

- Vary `--k` and watch where the scores fall off — that cliff hints at how many
  chunks are actually relevant vs. padding.
- Compare a precise query (`"minimum spanning tree algorithm"`) with a vague one
  (`"do stuff with nodes"`) and watch the scores drop.
- Note when several near-duplicate chunks come from the same file — that's the
  pain MMR-based retrieval would solve later.

## Generate (chat with the codebase)

The generation step turns retrieved chunks into plain-language answers using a
local Qwen model served by Ollama. All four front-ends share one RAG core
(`rag.py`) and one engine factory (`llm.py`).

### Set up the model in Ollama

Register the local GGUF with Ollama once (reuses the file, no re-download):

```sh
ollama create qwen2.5-coder-7b -f Modelfile.7b
```

`llm.py` defaults to this model, so no environment variables are needed. Optional
knobs: `OLLAMA_MODEL` (default `qwen2.5-coder-7b`), `LLM_TEMPERATURE` (default
0.2), `LLM_NUM_CTX` (default 16384).

> **Hardware note:** Qwen3-Coder-30B pushed past my 8 GB GPU / 16 GB free RAM, so
> the default is the smaller Qwen2.5-Coder-7B, which fits entirely in VRAM. The
> 30B `Modelfile` is still in the repo if your machine can handle it.

### The four front-ends

```bash
# 1. One-shot CLI — ask once, print answer, exit. No memory.
python generate.py "how do I export to graphviz"

# 2 & 3. Terminal chat — session memory + history-aware follow-ups.
python chat.py
python chat.py --no-history-aware      # disable rewriting to see the difference

# 4. Gradio web UI — browser chat at http://127.0.0.1:7860
python app.py
```

Every front-end grounds answers in retrieved code and cites the source files.
