from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from ingestion.connectors.base import BaseConnector


class PdfConnector(BaseConnector):
    """Loads PDF files from a directory using PyMuPDF.

    PyMuPDF correctly handles table text ordering and preserves page numbers,
    making it more reliable than pdfplumber or pypdf for enterprise documents.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(f"PDF directory not found: {self.directory}")

    def load(self) -> list[Document]:
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ImportError("PyMuPDF is required: pip install pymupdf") from e

        documents: list[Document] = []
        pdf_files = sorted(self.directory.rglob("*.pdf")) + sorted(
            self.directory.rglob("*.PDF")
        )

        for pdf_path in pdf_files:
            try:
                docs = self._load_pdf(fitz, pdf_path)
                documents.extend(docs)
            except Exception as exc:
                # Log but don't abort the whole ingestion run on a single bad file
                print(f"[PdfConnector] Warning: could not process {pdf_path}: {exc}")

        return documents

    def _load_pdf(self, fitz: object, pdf_path: Path) -> list[Document]:
        documents: list[Document] = []
        stat = pdf_path.stat()

        with fitz.open(str(pdf_path)) as doc:  # type: ignore[attr-defined]
            title = doc.metadata.get("title") or pdf_path.stem

            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if not text:
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(pdf_path.resolve()),
                            "source_type": "pdf",
                            "title": title,
                            "page": page_num,
                            "url": pdf_path.as_uri(),
                            "last_modified": self._iso_mtime(stat.st_mtime),
                        },
                    )
                )

        return documents

    @staticmethod
    def _iso_mtime(ts: float) -> str:
        from datetime import UTC, datetime

        return datetime.fromtimestamp(ts, tz=UTC).isoformat()
