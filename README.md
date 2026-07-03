# Explain it to me like I'm 5 — C# edition

A small pipeline for learning Retrieval-Augmented Generation (RAG) over a C#
codebase, running **fully offline**.

- **Embeddings:** `bge-small-en-v1.5` (local, in `models/`)
- **Vector store:** Chroma (persisted to `chroma_db/`)
- **Framework:** LangChain
- **Generation (later step):** `qwen3-coder` GGUF (already in `models/`)

## Pipeline steps

1. **Ingest** (this repo, done) — load C# files, chunk them C#-aware, embed with
   bge-small, store in Chroma.
2. **Retrieve** (done) — query the store for relevant code chunks.
3. **Generate** (done) — feed retrieved chunks + question to Qwen3-Coder for
   answers, via four front-ends (one-shot CLI, terminal chat, history-aware
   chat, Gradio web UI).

## Download offline models

```sh
pip install hf
hf download BAAI/bge-small-en-v1.5 --local-dir ./models/bge-small-en-v1.5
hf download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF --local-dir ./models/qwen3-coder-gguf --include "*Q4_K_M*"
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
git clone --depth 1 https://github.com/mathnet/mathnet-numerics.git repos/mathnet-numerics
git clone --depth 1 https://github.com/KeRNeLith/QuikGraph.git repos/quikgraph
```

## Ingest

```bash
python ingest.py --source repos/mathnet-numerics --reset
```

## Sanity-check the store

```bash
python ingest.py --query "how do I export to graphviz"
```

## Retrieve

`retrieve.py` embeds your query with the same offline bge-small model and prints
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
- Compare a precise query (`"Cholesky decomposition"`) with a vague one
  (`"do math"`) and watch the scores drop.
- Note when several near-duplicate chunks come from the same file — that's the
  pain MMR-based retrieval would solve later.

## Generate (chat with the codebase)

The generation step turns retrieved chunks into plain-language answers using the
local Qwen3-Coder model. All four front-ends share one RAG core (`rag.py`) and
one engine factory (`llm.py`).

### Pick an inference engine

You can check that the env is setup right:
```sh
where cl
```

Set `LLM_BACKEND` (default `ollama`):
```sh
# Option A: Ollama (default). Import your existing GGUF once, no re-download:
ollama create qwen3-coder -f Modelfile
SET LLM_BACKEND=ollama
```

If you use llama-cpp you have to setup the build chain.  Maybe just use Option A

Make sure you install vs build toold 2022 first from [here](https://aka.ms/vs/17/release/vs_buildtools.exe)
Then install "Desktop Development with C++".

Make sure your environment is setup. You may need to run
```
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

```sh
# Option B: llama-cpp-python, loads the GGUF in-process (no server):
pip install llama-cpp-python
SET LLM_BACKEND=llamacpp
```

Other env knobs: `LLM_TEMPERATURE` (default 0.2), `LLM_NUM_CTX` (default 16384),
`OLLAMA_MODEL` (default `qwen3-coder`), `LLM_N_GPU_LAYERS` (llama.cpp, default -1).

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
