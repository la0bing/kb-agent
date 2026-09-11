import os
import uuid

# Windows blocks symlink creation unless developer mode is on, which breaks the
# model downloads docling needs for PDF layout analysis
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import lancedb
import numpy as np
import pyarrow as pa
from docling.chunking import HybridChunker
from docling.datamodel.base_models import FormatToExtensions
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384
DB_PATH = os.environ.get("KB_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "kbdata"))
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# documents whose content is at least this similar are treated as the same subject
TOPIC_THRESHOLD = 0.80

# docling can read these, but audio/video also need the asr extra plus ffmpeg
SKIP_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".mp4", ".avi", ".mov", ".mkv", ".webm", ".tar.gz"}
SUPPORTED_EXTS = {"." + e for v in FormatToExtensions.values() for e in v} - SKIP_EXTS

CHUNK_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("source", pa.string()),
    pa.field("path", pa.string()),
    pa.field("text", pa.string()),
    pa.field("headings", pa.string()),
    pa.field("pages", pa.string()),
    pa.field("topic", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), DIM)),
])

TOPIC_SCHEMA = pa.schema([
    pa.field("name", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), DIM)),
])

db = lancedb.connect(DB_PATH)
encoder = None
chunker = None


def chunks():
    return db.create_table("chunks", schema=CHUNK_SCHEMA, exist_ok=True)


def topics():
    return db.create_table("topics", schema=TOPIC_SCHEMA, exist_ok=True)


def embed(texts):
    global encoder
    if encoder is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = SentenceTransformer(EMBED_MODEL, device=device)
    vectors = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def embed_query(text):
    return embed([QUERY_PREFIX + text])[0]


def supported(path):
    name = str(path).lower()
    return any(name.endswith(e) for e in SUPPORTED_EXTS)


def pick_topic(centroid, text):
    table = topics()
    known = table.search().limit(1000).to_list()
    if known:
        scores = np.array([r["vector"] for r in known], dtype=np.float32) @ centroid
        best = int(scores.argmax())
        if scores[best] >= TOPIC_THRESHOLD:
            return known[best]["name"]

    from kb_agent import qwen

    prompt = (
        "Give a short subject label of 2 to 4 words for the document below. "
        "Reply with the label only, no punctuation.\n\n"
    )
    label = qwen.ask(prompt + text[:1500], max_new_tokens=16).strip().strip("\"'.")
    label = label.splitlines()[0][:60] if label else ""
    if not label:
        return ""

    table.add([{"name": label, "vector": centroid.tolist()}])
    return label


def ingest_file(path, with_topics=True):
    global chunker
    if chunker is None:
        tokenizer = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL), max_tokens=512)
        chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)

    path = os.path.abspath(path)
    source = os.path.basename(path)
    doc = DocumentConverter().convert(path).document

    rows = []
    texts = []
    for chunk in chunker.chunk(dl_doc=doc):
        headings = " > ".join(chunk.meta.headings or [])
        pages = sorted({p.page_no for item in chunk.meta.doc_items for p in (item.prov or [])})
        rows.append({
            "id": uuid.uuid4().hex,
            "source": source,
            "path": path,
            "text": chunk.text,
            "headings": headings,
            "pages": ", ".join(str(p) for p in pages),
            "topic": "",
        })
        texts.append(chunker.contextualize(chunk=chunk))

    if not rows:
        return {"source": source, "chunks": 0, "topic": ""}

    vectors = embed(texts)
    centroid = vectors.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    topic = pick_topic(centroid, "\n".join(texts)) if with_topics else ""
    for row, vector in zip(rows, vectors):
        row["topic"] = topic
        row["vector"] = vector.tolist()

    table = chunks()
    table.delete("source = " + sql(source))
    table.add(rows)
    return {"source": source, "chunks": len(rows), "topic": topic}


def sql(value):
    return "'" + str(value).replace("'", "''") + "'"


def search(query, topic="", top_k=5):
    q = chunks().search(embed_query(query).tolist()).limit(max(1, top_k))
    if topic:
        q = q.where("topic = " + sql(topic))
    hits = []
    for row in q.to_list():
        hits.append({
            "id": row["id"],
            "source": row["source"],
            "headings": row["headings"],
            "pages": row["pages"],
            "topic": row["topic"],
            "text": row["text"],
            "score": round(1 - row["_distance"] / 2, 3),
        })
    return hits


def get(source=""):
    table = chunks()
    if not source:
        docs = {}
        for row in table.search().limit(100000).select(["source", "topic"]).to_list():
            key = row["source"]
            if key not in docs:
                docs[key] = {"source": key, "topic": row["topic"], "chunks": 0}
            docs[key]["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["source"])

    rows = table.search().where("source = " + sql(source)).limit(100000).to_list()
    return [{k: r[k] for k in ("id", "source", "headings", "pages", "topic", "text")} for r in rows]


def edit(chunk_id, new_text):
    table = chunks()
    where = "id = " + sql(chunk_id)
    if not table.search().where(where).limit(1).to_list():
        return 0
    table.update(where=where, values={"text": new_text, "vector": embed([new_text])[0].tolist()})
    return 1


def delete(source):
    table = chunks()
    where = "source = " + sql(source)
    removed = table.count_rows(where)
    if removed:
        table.delete(where)
    return removed
