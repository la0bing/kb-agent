# kb-agent

A local knowledge base agent. Documents are parsed with docling, embedded with
BAAI/bge-small-en-v1.5 and stored in LanceDB. A Google ADK agent running
Qwen3.5-2B answers questions over them and manages the store through tools.

Everything runs on your machine. No API keys, no calls off the box.

## Setup

### 1. Install

```
pip install -r requirements.txt
cp .env.example .env
```

Both models download from Hugging Face on first use (Qwen3.5-2B is about
4.3 GB, bge-small about 130 MB).

### 2. Start the LLM — terminal 1

```
transformers serve Qwen/Qwen3.5-2B --dtype bfloat16 --reasoning off
```

Leave this running. Check it with `curl http://localhost:8000/v1/models`.

Passing the model as a positional argument preloads it so it is never unloaded
between questions, and `--reasoning off` drops Qwen3.5's thinking traces for
faster replies. Short on VRAM? Add `--quantization bnb-4bit`.

### 3. Ingest your documents — terminal 2

```
python ingest.py
```

With no argument it reads `docs/`; pass a path to point it at another folder
or a single file. This needs terminal 1 already running, because it asks the
model to name each new topic — add `--no-topics` to skip that and run it
without the server.

### 4. Start the chat UI — terminal 2

```
adk web --port 8080
```

Open the URL it prints and pick `kb_agent`. Port 8080 because `transformers
serve` already has 8000. `adk run kb_agent` gives the same agent in the
terminal instead.

## Ingesting

```
python ingest.py                          # docs/, recursively
python ingest.py C:\my\documents          # any other folder
python ingest.py report.pdf               # one file
python ingest.py C:\my\docs --no-topics   # skip topic grouping
```

Every format docling supports is accepted (pdf, docx, pptx, xlsx, html, md,
csv, epub, images and more) except audio and video, which need the extra
`docling[asr]` install and ffmpeg. Re-ingesting a file replaces its old chunks
rather than duplicating them.

You can also just ask the agent in chat: "ingest C:\my\documents". It calls the
same code as the script, so both behave identically.

## Tools

| tool | what it does |
| --- | --- |
| `search_kb` | finds the passages most relevant to a question, optionally within one topic |
| `get_document` | lists the stored documents, or the chunks of one of them |
| `edit_document` | rewrites a chunk and re-embeds it |
| `delete_document` | removes a document and all of its chunks |
| `ingest_path` | adds a file or folder, using the same code path as `ingest.py` |

## Demo

Three short sessions in `adk web`. The repo ships one document in `evals/docs`,
so they work on a fresh checkout with nothing of your own ingested.

### 1. Ask the knowledge base

| prompt | what to expect |
| --- | --- |
| `ingest C:\path\to\kb-agent\evals\docs` | calls `ingest_path`, reports `student_handbook.md` and 5 chunks |
| `what is the minimum attendance to sit for the final exam?` | calls `search_kb`, answers 80% and names `student_handbook.md` |
| `what happens if my CGPA drops below 2.00?` | academic probation and a meeting with an advisor, from the same source |

The whole loop: docling parses the file, the chunks are embedded, and the
answer comes back with the source named rather than from the model's memory.

### 2. It will not guess

| prompt | what to expect |
| --- | --- |
| `how do I claim mileage for driving to work?` | nothing scores above `MIN_SCORE`, so it says it found nothing |
| `what documents do you have?` | calls `get_document`, lists the handbook and its topic |

The handbook says nothing about mileage. Without the score threshold the
search would still return five passages and a 2B model would try to answer
from them, so this is the prompt worth showing.

### 3. Fix a wrong fact

| prompt | what to expect |
| --- | --- |
| `show me the chunks in student_handbook.md` | calls `get_document`, lists all 5 chunks with their ids |
| `change the attendance requirement to 75% in chunk <id>` | calls `edit_document`, confirms the chunk was updated |
| `what is the minimum attendance now?` | answers 75% |

The last prompt is the point. A stored edit that was not re-embedded would
still answer 80%, so asking again is what proves the new text is searchable.
`delete student_handbook.md` clears it afterwards.

## Topics

Ingestion groups documents into subjects on its own. Each document's chunk
vectors are averaged into one centroid, and if that centroid is at least 80%
similar to a topic already stored the document joins it. Otherwise the model is
asked for a short label and a new topic starts. Nobody has to name anything,
and the model is only called for genuinely new subjects.

Once grouped you can narrow a search to one subject, either by asking the agent
or directly:

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

`adk web` needs the agent in a package directory whose name is a valid Python
identifier, which is why `kb_agent/` exists rather than the files sitting at
the repo root.

## Notes

The agent talks to the model over an OpenAI-compatible API, so anything
speaking that protocol works — point `LLM_API_BASE` at Ollama, llama.cpp or
vLLM and nothing else changes. Settings live in `.env`:

```
LLM_MODEL=Qwen/Qwen3.5-2B
LLM_API_BASE=http://localhost:8000/v1
LLM_API_KEY=not-needed
```

A 2B model hands tool arguments back as strings even when the schema says
integer, so `store.search` casts `top_k` itself.

Search drops anything scoring below `store.MIN_SCORE` (0.60) rather than
handing weak passages to the model, which is what lets it say it found nothing.
Raise or lower it once you have enough documents to see where real answers
land — `python evals/test_retrieval.py` prints the scores.

Set `KB_DB_PATH` to put the database somewhere other than `./kbdata`.
