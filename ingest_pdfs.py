"""Ingest authorized PDFs from data/source_pdfs into the local RAG knowledge base."""

import argparse
import json
from pathlib import Path

from tools.pdf_ingestion import OUTPUT_PATH, SOURCE_PDF_PATH, ingest_pdfs


def main():
    parser = argparse.ArgumentParser(description="Extract authorized PDFs for local healthcare RAG.")
    parser.add_argument("--source", type=Path, default=SOURCE_PDF_PATH, help="Folder containing PDFs.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Folder for extracted JSON.")
    parser.add_argument("--force", action="store_true", help="Re-ingest PDFs that already have output JSON.")
    arguments = parser.parse_args()
    summary = ingest_pdfs(arguments.source, arguments.output, arguments.force)
    print(json.dumps(summary, indent=2))
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
