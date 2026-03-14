"""
Multi-modal document processor: text, tables, OCR from PDFs and images.
"""
import io
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import structlog
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

logger = structlog.get_logger()

# Optional imports with graceful fallback
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available - table extraction disabled")

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("pytesseract/PIL not available - OCR disabled")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


class DocumentProcessor:
    """
    Handles multi-modal document processing:
    - Text extraction from PDFs and text files
    - Table extraction using pdfplumber
    - OCR for scanned documents and images
    - Chunking with configurable overlap
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def process_pdf(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a PDF file with text + table + OCR extraction.
        Returns list of chunks with metadata.
        """
        metadata = metadata or {}
        chunks = []

        logger.info("Processing PDF", path=file_path)

        # 1. Text extraction via LangChain PyPDFLoader
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            for page in pages:
                page_meta = {
                    **metadata,
                    "page": page.metadata.get("page", 0) + 1,
                    "extraction_method": "text",
                }
                chunks.extend(
                    self._chunk_text(page.page_content, page_meta)
                )
        except Exception as e:
            logger.error("PDF text extraction failed", error=str(e))

        # 2. Table extraction via pdfplumber
        if PDFPLUMBER_AVAILABLE:
            table_chunks = self._extract_tables_pdf(file_path, metadata)
            chunks.extend(table_chunks)

        # 3. OCR fallback for scanned pages
        if PDF2IMAGE_AVAILABLE and OCR_AVAILABLE:
            ocr_chunks = self._ocr_pdf(file_path, metadata)
            chunks.extend(ocr_chunks)

        logger.info("PDF processing complete", chunks=len(chunks))
        return chunks

    def process_image(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """OCR an image file and return chunks."""
        if not OCR_AVAILABLE:
            logger.warning("OCR not available, skipping image", path=file_path)
            return []

        metadata = metadata or {}
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            if text.strip():
                meta = {**metadata, "extraction_method": "ocr", "source_file": file_path}
                return self._chunk_text(text, meta)
        except Exception as e:
            logger.error("Image OCR failed", path=file_path, error=str(e))
        return []

    def process_text(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Process plain text or markdown files."""
        metadata = metadata or {}
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            chunks = []
            for doc in docs:
                meta = {**metadata, "extraction_method": "text", "source_file": file_path}
                chunks.extend(self._chunk_text(doc.page_content, meta))
            return chunks
        except Exception as e:
            logger.error("Text processing failed", path=file_path, error=str(e))
            return []

    def process_raw_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Process raw text string directly."""
        metadata = metadata or {}
        return self._chunk_text(text, metadata)

    def _extract_tables_pdf(
        self, file_path: str, metadata: Dict
    ) -> List[Dict[str, Any]]:
        """Extract tables from PDF using pdfplumber."""
        chunks = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        table_text = self._table_to_text(table)
                        if table_text.strip():
                            meta = {
                                **metadata,
                                "page": page_num,
                                "table_index": table_idx,
                                "extraction_method": "table",
                                "content_type": "table",
                            }
                            chunks.extend(self._chunk_text(table_text, meta))
        except Exception as e:
            logger.error("Table extraction failed", error=str(e))
        return chunks

    def _ocr_pdf(self, file_path: str, metadata: Dict) -> List[Dict[str, Any]]:
        """OCR a PDF by converting to images first."""
        chunks = []
        try:
            images = convert_from_path(file_path, dpi=200)
            for page_num, image in enumerate(images, 1):
                text = pytesseract.image_to_string(image)
                if len(text.strip()) > 100:  # Only if substantial text found
                    meta = {
                        **metadata,
                        "page": page_num,
                        "extraction_method": "ocr",
                    }
                    chunks.extend(self._chunk_text(text, meta))
        except Exception as e:
            logger.error("PDF OCR failed", error=str(e))
        return chunks

    def _table_to_text(self, table: List[List]) -> str:
        """Convert table rows to readable text."""
        if not table:
            return ""
        lines = []
        header = table[0] if table else []
        header_clean = [str(h or "").strip() for h in header]
        lines.append(" | ".join(header_clean))
        lines.append("-" * 60)
        for row in table[1:]:
            row_clean = [str(c or "").strip() for c in row]
            lines.append(" | ".join(row_clean))
        return "\n".join(lines)

    def _chunk_text(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Split text into chunks and attach metadata."""
        if not text or not text.strip():
            return []
        text = self._clean_text(text)
        splits = self.text_splitter.split_text(text)
        return [
            {"text": chunk, "metadata": {**metadata, "chunk_index": i}}
            for i, chunk in enumerate(splits)
            if chunk.strip()
        ]

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text.strip()

    def auto_detect_source(self, filename: str) -> str:
        """Guess source type from filename."""
        name = filename.lower()
        if "who" in name:
            return "WHO Guidelines"
        elif "pubmed" in name or "ncbi" in name:
            return "PubMed"
        elif "lab" in name or "report" in name or "blood" in name:
            return "Lab Report"
        elif "textbook" in name or "manual" in name:
            return "Medical Textbook"
        elif "clinical" in name or "trial" in name or "research" in name:
            return "Clinical Research"
        return "Other"


document_processor = DocumentProcessor()