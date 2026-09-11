import os

from google.adk.agents import Agent

from kb_agent import store
from kb_agent.qwen import QwenLocal


def search_kb(query: str, topic: str, top_k: int) -> dict:
    """Search the knowledge base for passages relevant to a question.

    Args:
        query: What to look for, in natural language.
        topic: Restrict results to this topic name. Pass an empty string for no filter.
        top_k: How many passages to return. Pass 5 if unsure.
    """
    hits = store.search(query, topic, top_k or 5)
    if not hits:
        return {"status": "empty", "message": "Nothing relevant was found in the knowledge base."}
    return {"status": "success", "results": hits}


def get_document(source: str) -> dict:
    """List the documents in the knowledge base, or show the chunks of one document.

    Args:
        source: The file name of the document to inspect, for example report.pdf.
            Pass an empty string to list every document instead.
    """
    rows = store.get(source)
    if not rows:
        return {"status": "empty", "message": "The knowledge base has no matching documents."}
    return {"status": "success", "documents" if not source else "chunks": rows}


def edit_document(chunk_id: str, new_text: str) -> dict:
    """Replace the text of one stored chunk and re-embed it.

    Args:
        chunk_id: The id of the chunk, as returned by search_kb or get_document.
        new_text: The replacement text.
    """
    if store.edit(chunk_id, new_text):
        return {"status": "success", "message": "Chunk " + chunk_id + " was updated."}
    return {"status": "error", "message": "No chunk with id " + chunk_id + " exists."}


def delete_document(source: str) -> dict:
    """Delete a document and all of its chunks from the knowledge base.

    Args:
        source: The file name of the document to delete, for example report.pdf.
    """
    removed = store.delete(source)
    if not removed:
        return {"status": "error", "message": "No document named " + source + " is stored."}
    return {"status": "success", "message": "Deleted " + source + " and " + str(removed) + " chunks."}


def ingest_path(path: str) -> dict:
    """Parse, chunk and add a file or a whole directory to the knowledge base.

    Args:
        path: Absolute path to a file or a directory of files to ingest.
    """
    if not os.path.exists(path):
        return {"status": "error", "message": path + " does not exist."}

    targets = []
    if os.path.isfile(path):
        targets.append(path)
    else:
        for root, _, names in os.walk(path):
            targets.extend(os.path.join(root, n) for n in names if store.supported(n))

    targets = [t for t in targets if store.supported(t)]
    if not targets:
        return {"status": "error", "message": "No files docling can read were found at " + path + "."}

    done, failed = [], []
    for target in targets:
        try:
            done.append(store.ingest_file(target))
        except Exception as error:
            failed.append({"source": os.path.basename(target), "error": str(error)})

    return {"status": "success" if done else "error", "ingested": done, "failed": failed}


root_agent = Agent(
    name="kb_agent",
    model=QwenLocal(),
    description="Answers questions from a local knowledge base of ingested documents.",
    instruction=(
        "You are a knowledge base assistant.\n"
        "Always call search_kb before answering a question about the documents, and base your "
        "answer only on what it returns. Name the source file you took the answer from. "
        "If the search returns nothing, say so plainly instead of guessing.\n"
        "Use get_document to list what is stored, ingest_path to add new files, "
        "edit_document to correct a chunk, and delete_document to remove a document.\n"
        "Call one tool at a time and keep answers short."
    ),
    tools=[search_kb, get_document, edit_document, delete_document, ingest_path],
)
