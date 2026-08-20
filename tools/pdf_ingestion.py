"""Controlled ingestion of locally downloaded, authorized PDFs for RAG."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pypdf import PdfReader


SOURCE_PDF_PATH = Path("data/source_pdfs")
OUTPUT_PATH = Path("data/knowledge_base/pdfs")


def _safe_source_id(pdf_path: Path) -> str:
    name = "".join(
        character.lower() if character.isalnum() else "_"
        for character in pdf_path.stem
    ).strip("_")
    return name or "document"


def _file_sha256(pdf_path: Path) -> str:
    digest = hashlib.sha256()
    with pdf_path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_sidecar_metadata(pdf_path: Path) -> Dict[str, Any]:
    """Read optional `<filename>.metadata.json` source metadata."""
    sidecar = pdf_path.with_suffix(".metadata.json")
    if not sidecar.exists():
        return {}
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Metadata sidecar must be a JSON object: {sidecar}")
    return data


def extract_pdf(pdf_path: Path) -> Tuple[List[Dict[str, Any]], int]:
    """Extract non-empty pages as independently citable RAG documents."""
    metadata = _load_sidecar_metadata(pdf_path)
    reader = PdfReader(str(pdf_path))
    source_id = metadata.get("source_id", _safe_source_id(pdf_path))
    extracted_at = datetime.now(timezone.utc).isoformat()
    file_hash = _file_sha256(pdf_path)
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        documents.append({
            "source_id": f"{source_id}_page_{page_number}",
            "source_type": metadata.get("source_type", "authorized_pdf"),
            "organization": metadata.get("organization", ""),
            "title": metadata.get("title", pdf_path.stem),
            "publication_date": metadata.get("publication_date", ""),
            "document_version": metadata.get("document_version", ""),
            "url": metadata.get("url", ""),
            "pmid": str(metadata.get("pmid", "")),
            "doi": metadata.get("doi", ""),
            "section": f"Page {page_number}",
            "content": text,
            "source_file": pdf_path.name,
            "source_sha256": file_hash,
            "ingested_at": extracted_at,
        })
    return documents, len(reader.pages)


def ingest_pdfs(source_path: Path = SOURCE_PDF_PATH, output_path: Path = OUTPUT_PATH, force: bool = False) -> Dict[str, Any]:
    """Convert each PDF into a JSON document collection used by `tools.rag`."""
    source_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    summary = {"processed": [], "skipped": [], "errors": []}

    for pdf_path in sorted(source_path.glob("*.pdf")):
        output_file = output_path / f"{_safe_source_id(pdf_path)}.json"
        if output_file.exists() and not force:
            summary["skipped"].append({"file": pdf_path.name, "reason": "already ingested"})
            continue
        try:
            documents, total_pages = extract_pdf(pdf_path)
            if not documents:
                summary["errors"].append({
                    "file": pdf_path.name,
                    "reason": "No extractable text. This may be a scanned PDF; OCR is required.",
                })
                continue
            output_file.write_text(json.dumps(documents, indent=2, ensure_ascii=False), encoding="utf-8")
            summary["processed"].append({
                "file": pdf_path.name,
                "pages": total_pages,
                "extractable_pages": len(documents),
                "output": str(output_file),
            })
        except Exception as error:
            summary["errors"].append({"file": pdf_path.name, "reason": str(error)})
    return summary
