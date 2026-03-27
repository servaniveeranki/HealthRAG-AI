"""
Feature 4: Automatic Document Ingestion Script
=============================================
Adds medical PDFs or text files into ChromaDB automatically:
  - splits into chunks
  - generates embeddings
  - stores in ChromaDB

Usage examples:
  # Single file
  python ingest.py --file path/to/guideline.pdf

  # Entire folder
  python ingest.py --folder path/to/docs/

  # Raw text with metadata
  python ingest.py --text "Aspirin 75mg is used for..." --source "BNF" --title "Aspirin Monograph"

  # Watch a folder and auto-ingest new files
  python ingest.py --watch path/to/docs/
"""
import argparse
import os
import sys
import time
import uuid
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# ── Bootstrap: make sure app/ is importable ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(20))
logger = structlog.get_logger()

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".text"}
CHUNK_SIZE    = 500   # characters
CHUNK_OVERLAP = 80


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks, respecting sentence boundaries."""
    text = text.strip()
    if not text:
        return []

    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to end at a sentence boundary
        if end < len(text):
            for boundary in (". ", ".\n", "? ", "! ", "\n\n"):
                pos = text.rfind(boundary, start + chunk_size // 2, end)
                if pos != -1:
                    end = pos + len(boundary)
                    break

        chunk = text[start:end].strip()
        if len(chunk) > 40:          # skip tiny fragments
            chunks.append(chunk)
        start = end - overlap

    return chunks


# ── Text extraction ───────────────────────────────────────────────────────────
def extract_text_from_pdf(path: str) -> str:
    """Extract text from PDF using pypdf (fallback: pdfplumber)."""
    text = ""
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
    except Exception as e:
        logger.warning("pypdf failed, trying pdfplumber", error=str(e))
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
        except Exception as e2:
            logger.error("PDF extraction failed", path=path, error=str(e2))
    return text.strip()


def extract_text(path: str) -> str:
    """Extract text from a file based on its extension."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    else:  # .txt, .md, .text
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()


# ── Ingestion core ────────────────────────────────────────────────────────────
def ingest_document(
    text: str,
    source: str,
    title: str,
    source_type: str = "Other",
    document_date: Optional[str] = None,
    guideline_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Chunk text, embed, and store in ChromaDB.
    Returns a summary dict.
    """
    from app.core.vector_store import vector_store

    if not text.strip():
        return {"success": False, "reason": "Empty text", "chunks": 0}

    document_date = document_date or datetime.now().strftime("%Y-%m-%d")
    doc_id = hashlib.md5(f"{title}{source}{text[:200]}".encode()).hexdigest()[:12]

    chunks = chunk_text(text)
    if not chunks:
        return {"success": False, "reason": "No chunks produced", "chunks": 0}

    chunk_dicts = []
    for i, chunk in enumerate(chunks):
        chunk_dicts.append({
            "id":   f"{doc_id}_chunk_{i}",
            "text": chunk,
            "metadata": {
                "source":            source,
                "title":             title,
                "organization":      source_type,
                "source_type":       source_type,
                "document_date":     document_date,
                "guideline_version": guideline_version or "",
                "page":              i + 1,
                "doc_id":            doc_id,
            },
        })

    # Use vector_store.add_documents which handles embedding internally
    ids = vector_store.add_documents(chunk_dicts)
    logger.info("Ingested document", title=title, chunks=len(ids), doc_id=doc_id)
    return {"success": True, "doc_id": doc_id, "chunks": len(ids), "title": title}


def ingest_file(
    file_path: str,
    source_type: str = "Other",
    document_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Ingest a single file."""
    path = Path(file_path)
    if not path.exists():
        return {"success": False, "reason": f"File not found: {file_path}", "chunks": 0}
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return {"success": False, "reason": f"Unsupported type: {path.suffix}", "chunks": 0}

    logger.info("Ingesting file", path=str(path))
    text = extract_text(str(path))
    if not text:
        return {"success": False, "reason": "Could not extract text", "chunks": 0}

    return ingest_document(
        text=text,
        source=path.stem.replace("_", " ").replace("-", " ").title(),
        title=path.stem.replace("_", " ").replace("-", " ").title(),
        source_type=source_type,
        document_date=document_date,
    )


def ingest_folder(
    folder_path: str,
    source_type: str = "Other",
    document_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Ingest all supported files in a folder."""
    folder = Path(folder_path)
    if not folder.is_dir():
        logger.error("Not a directory", path=folder_path)
        return []

    results = []
    files = [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    logger.info("Found files to ingest", count=len(files), folder=folder_path)

    for f in files:
        result = ingest_file(str(f), source_type, document_date)
        results.append(result)
        status = "✅" if result["success"] else "❌"
        print(f"  {status} {f.name} → {result.get('chunks', 0)} chunks")

    return results


def watch_folder(folder_path: str, source_type: str = "Other", poll_interval: int = 10):
    """Watch a folder and ingest new files as they appear."""
    folder = Path(folder_path)
    seen   = set(str(f) for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)
    logger.info("Watching folder", path=folder_path, existing_files=len(seen))
    print(f"\n👁️  Watching {folder_path} for new files (Ctrl+C to stop)...\n")

    try:
        while True:
            time.sleep(poll_interval)
            current = set(str(f) for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)
            new_files = current - seen
            for fp in new_files:
                print(f"🆕 New file detected: {Path(fp).name}")
                result = ingest_file(fp, source_type)
                status = "✅" if result["success"] else "❌"
                print(f"   {status} Ingested → {result.get('chunks', 0)} chunks")
            seen = current
    except KeyboardInterrupt:
        print("\n⏹️  Stopped watching.")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Medical RAG — Automatic Document Ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py --file guidelines/who_diabetes_2024.pdf
  python ingest.py --folder docs/ --source-type "WHO Guidelines"
  python ingest.py --text "Metformin is first-line..." --source "BNF" --title "Metformin Entry"
  python ingest.py --watch docs/ --poll 15
        """,
    )
    parser.add_argument("--file",        help="Path to a single PDF or text file")
    parser.add_argument("--folder",      help="Path to a folder of documents")
    parser.add_argument("--text",        help="Raw text string to ingest directly")
    parser.add_argument("--watch",       help="Folder to watch for new files (continuous)")
    parser.add_argument("--source",      default="Medical Document", help="Source name")
    parser.add_argument("--title",       default="Untitled Document",help="Document title")
    parser.add_argument("--source-type", default="Other",
                        choices=["WHO Guidelines","PubMed","Medical Textbook",
                                 "Clinical Research","Lab Report","Other"],
                        help="Document source type")
    parser.add_argument("--date",        help="Document date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--poll",        type=int, default=10,
                        help="Poll interval in seconds for --watch mode")

    args = parser.parse_args()

    if not any([args.file, args.folder, args.text, args.watch]):
        parser.print_help()
        sys.exit(0)

    print("\n🏥 Medical RAG — Document Ingestion\n" + "─" * 40)

    if args.file:
        result = ingest_file(args.file, args.source_type, args.date)
        if result["success"]:
            print(f"✅ Ingested '{result['title']}' → {result['chunks']} chunks (ID: {result['doc_id']})")
        else:
            print(f"❌ Failed: {result.get('reason', 'unknown error')}")

    elif args.folder:
        results = ingest_folder(args.folder, args.source_type, args.date)
        ok  = sum(1 for r in results if r["success"])
        tot = len(results)
        print(f"\n✅ Done: {ok}/{tot} files ingested successfully.")

    elif args.text:
        result = ingest_document(
            text=args.text,
            source=args.source,
            title=args.title,
            source_type=args.source_type,
            document_date=args.date,
        )
        if result["success"]:
            print(f"✅ Ingested '{result['title']}' → {result['chunks']} chunks")
        else:
            print(f"❌ Failed: {result.get('reason')}")

    elif args.watch:
        watch_folder(args.watch, args.source_type, args.poll)


if __name__ == "__main__":
    main()