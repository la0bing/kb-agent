# kb-agent

A local knowledge base agent. Documents are parsed with docling, embedded with
BAAI/bge-small-en-v1.5 and stored in LanceDB. A Google ADK agent running
Qwen3.5-2B answers questions over them and manages the store through tools.

Everything runs on your machine. No API keys, no calls off the box.

## Setup

```
pip install -r requirements.txt
cp .env.example .env
```

The two models are pulled from Hugging Face the first time they are used
(Qwen3.5-2B is about 4.3 GB, bge-small is about 130 MB).

## Running

Two terminals. `transformers serve` and `adk web` both default to port 8000, so
the chat UI is moved to 8080.

### Terminal 1 - the LLM

```
transformers serve Qwen/Qwen3.5-2B --dtype bfloat16 --reasoning off
```

Passing the model as a positional argument preloads it and keeps it in memory
so it is never unloaded between questions. `--reasoning off` turns off
Qwen3.5's thinking traces, which makes replies faster and the trace view
cleaner. Short on VRAM? Add `--quantization bnb-4bit`.

Check it is up with `curl http://localhost:8000/v1/models`.

### Terminal 2 - the agent

```
adk web --port 8080
```

Open the URL it prints and pick `kb_agent`. `adk run kb_agent` gives the same
agent in the terminal.

## Ingesting

Point it at a folder or a single file:

```
python ingest.py C:\my\documents
python ingest.py report.pdf
python ingest.py C:\my\documents --no-topics
```

Every format docling supports is accepted (pdf, docx, pptx, xlsx, html, md,
csv, epub, images and more) except audio and video, which need the extra
`docling[asr]` install and ffmpeg. Re-ingesting a file replaces its old chunks
rather than duplicating them.

## Tools

The agent has five tools:

| tool | what it does |
| --- | --- |
| `search_kb` | finds the passages most relevant to a question, optionally within one topic |
| `get_document` | lists the stored documents, or the chunks of one of them |
| `edit_document` | rewrites a chunk and re-embeds it |
| `delete_document` | removes a document and all of its chunks |
| `ingest_path` | adds a file or folder, using the same code path as `ingest.py` |

Because `ingest_path` calls the same `store.ingest_file` that the script uses,
ingesting from chat and ingesting from the command line behave identically.

## Topics

Ingestion groups documents into subjects on its own. Each document's chunk
vectors are averaged into a single centroid, and if that centroid is at least
80% similar to a topic already in the store the document joins it. Otherwise
the model is asked for a short label and a new topic is created. Nobody has to
name anything, and the model is only called when a genuinely new subject shows
up. Pass `--no-topics` to skip it.

Once documents are grouped you can narrow a search to one subject, either by
asking the agent or directly:

```python
from kb_agent import store
store.search("carry over rules", "Paid Time Off Rules", 5)
```

## Layout

```
kb_agent/
  agent.py    the five tools and the agent definition
  llm.py      which model to talk to and where it is served
  store.py    docling parsing, embeddings and all LanceDB access
ingest.py     the bulk ingestion script
kbdata/       the LanceDB database, created on first use
```

`adk web` needs the agent to live in a package directory whose name is a valid
Python identifier, which is why `kb_agent/` exists rather than the files
sitting at the repo root.

## Notes

The agent talks to the model over an OpenAI-compatible API, so anything that
speaks that protocol works. Point `LLM_API_BASE` at Ollama, llama.cpp or vLLM
instead and nothing else has to change. The settings live in `.env`:

```
LLM_MODEL=Qwen/Qwen3.5-2B
LLM_API_BASE=http://localhost:8000/v1
LLM_API_KEY=not-needed
```

A 2B model hands tool arguments back as strings even when the schema says
integer, so `store.search` casts `top_k` itself.

Set `KB_DB_PATH` to put the database somewhere other than `./kbdata`.
