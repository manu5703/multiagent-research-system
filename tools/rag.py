"""Local, provenance-preserving retrieval for approved healthcare evidence.

Uses TF-IDF from scikit-learn, which is already a dependency. The public
functions can later be backed by an embedding/vector store without changing the
agents.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


KNOWLEDGE_BASE_PATH = Path("data/knowledge_base")
CHUNK_SIZE = 300


def _normalise_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text on sentence boundaries while retaining useful context."""
    text = _normalise_text(text)
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, length = [], [], 0
    for sentence in sentences:
        if current and length + len(sentence) + 1 > chunk_size:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(sentence)
        length += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def _document_chunks(document: Dict[str, Any], index: int) -> List[Dict[str, Any]]:
    text = _normalise_text(document.get("content") or document.get("text"))
    if not text:
        return []
    source_id = _normalise_text(document.get("source_id")) or f"local_{index}"
    metadata = {
        "source_type": document.get("source_type", "approved_local_document"),
        "organization": document.get("organization", ""),
        "title": document.get("title", source_id),
        "url": document.get("url", ""),
        "pmid": str(document.get("pmid", "")),
        "doi": document.get("doi", ""),
        "publication_date": document.get("publication_date", ""),
        "document_version": document.get("document_version", ""),
        "section": document.get("section", ""),
    }
    return [
        {"evidence_id": f"{source_id}_chunk_{part}", "content": content, **metadata}
        for part, content in enumerate(chunk_text(text), start=1)
    ]


def load_approved_documents(path: Path = KNOWLEDGE_BASE_PATH) -> List[Dict[str, Any]]:
    """Load approved .json, .jsonl, and .txt documents from the local KB.

    JSON records need ``content`` or ``text`` and should include source
    provenance. Do not place PHI or unapproved documents in this directory.
    """
    if not path.exists():
        return []
    documents = []
    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix == ".txt":
            documents.append({"source_id": file_path.stem, "title": file_path.stem.replace("_", " "), "content": file_path.read_text(encoding="utf-8")})
        elif file_path.suffix == ".json":
            data = json.loads(file_path.read_text(encoding="utf-8"))
            documents.extend(data if isinstance(data, list) else [data])
        elif file_path.suffix == ".jsonl":
            documents.extend(json.loads(line) for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return documents


def pubmed_documents(articles: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert fetched PubMed article abstracts into citable RAG documents."""
    documents = []
    for article in articles:
        abstract = _normalise_text(article.get("abstract"))
        if abstract:
            pmid = str(article.get("pmid", ""))
            documents.append({
                "source_id": f"pmid_{pmid}", "source_type": "pubmed_abstract",
                "title": article.get("title", "PubMed article"), "content": abstract,
                "pmid": pmid, "publication_date": article.get("year", ""),
                "organization": "PubMed", "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            })
    return documents


def retrieve_evidence(query: str, documents: Iterable[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
    """Return relevant chunks with stable source provenance."""
    chunks = [chunk for index, document in enumerate(documents) for chunk in _document_chunks(document, index)]
    if not chunks or not _normalise_text(query):
        return []
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([chunk["content"] for chunk in chunks])
    scores = cosine_similarity(vectorizer.transform([query]), matrix).ravel()
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
    retrieved_at = datetime.now(timezone.utc).isoformat()
    return [{**chunks[index], "relevance_score": float(score), "retrieved_at": retrieved_at} for index, score in ranked if score > 0]
