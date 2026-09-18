# kb-agent

A local knowledge base agent. Documents are parsed with docling, embedded with
BAAI/bge-small-en-v1.5 and stored in LanceDB. A Google ADK agent running
Qwen3.5-2B answers questions over them. Everything runs on your machine.

## Setup

```
pip install -r requirements.txt
cp .env.example .env
```

Both models download from Hugging Face on first use (about 4.4 GB total).

## Run

### 1. Start the LLM — terminal 1

```
transformers serve Qwen/Qwen3.5-2B --dtype bfloat16 --reasoning off
```

Leave it running. Check with `curl http://localhost:8000/v1/models`.

### 2. Ingest documents — terminal 2

```
python ingest.py
```

Reads `docs/`, or pass a folder or file instead.

### 3. Start the chat UI — terminal 2

```
adk web --port 8080
```

Open the URL it prints and pick `kb_agent`.

## Demo

Three sessions in `adk web`, using the document in `evals/docs`.

### 1. Ask the knowledge base

| prompt | what to expect |
| --- | --- |
| `ingest C:\path\to\kb-agent\evals\docs` | reports `student_handbook.md` and 5 chunks |
| `what is the minimum attendance to sit for the final exam?` | answers 80% and names `student_handbook.md` |
| `what happens if my CGPA drops below 2.00?` | academic probation and a meeting with an advisor |

### 2. It will not guess

| prompt | what to expect |
| --- | --- |
| `how do I claim mileage for driving to work?` | nothing scores high enough, so it says it found nothing |
| `what documents do you have?` | lists the handbook and its topic |

### 3. Fix a wrong fact

| prompt | what to expect |
| --- | --- |
| `show me the chunks in student_handbook.md` | lists all 5 chunks with their ids |
| `change the attendance requirement to 75% in chunk <id>` | confirms the chunk was updated |
| `what is the minimum attendance now?` | answers 75%, proving the edit was re-embedded |

`delete student_handbook.md` clears it afterwards.
