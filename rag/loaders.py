"""Document loaders: PDF, DOCX, TXT/MD, CSV and XLSX.

Every loader returns a list of `Page` objects. A "page" is the smallest unit we
chunk from - a PDF page, a DOCX section, a block of spreadsheet rows, etc.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config


@dataclass
class Page:
    text: str
    source: str            # file name
    path: str              # absolute path
    page: int | None = None
    section: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 256), b""):
            h.update(block)
    return h.hexdigest()


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def _load_pdf_pdfium(path: Path) -> list[Page]:
    """Primary engine: PDFium handles large/complex PDFs pypdf chokes on."""
    import pypdfium2 as pdfium

    pages: list[Page] = []
    document = pdfium.PdfDocument(str(path))
    try:
        for index in range(len(document)):
            page = document[index]
            try:
                textpage = page.get_textpage()
                try:
                    raw = textpage.get_text_range()
                finally:
                    textpage.close()
            finally:
                page.close()
            text = clean_text(raw or "")
            if len(text) < 20:
                continue
            pages.append(Page(text=text, source=path.name, path=str(path), page=index + 1))
    finally:
        document.close()
    return pages


def _load_pdf_pypdf(path: Path) -> list[Page]:
    """Fallback engine."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[Page] = []
    for index, pdf_page in enumerate(reader.pages, start=1):
        try:
            raw = pdf_page.extract_text() or ""
        except Exception:
            raw = ""
        text = clean_text(raw)
        if len(text) < 20:
            continue
        pages.append(Page(text=text, source=path.name, path=str(path), page=index))
    return pages


def load_pdf(path: Path) -> list[Page]:
    try:
        pages = _load_pdf_pdfium(path)
        if pages:
            return pages
    except Exception:
        pass
    return _load_pdf_pypdf(path)


# --------------------------------------------------------------------------- #
# DOCX (heading aware)
# --------------------------------------------------------------------------- #
def load_docx(path: Path) -> list[Page]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    blocks: list[tuple[str | None, list[str]]] = []
    current_heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        text = clean_text("\n".join(buffer))
        if text:
            blocks.append((current_heading, [text]))
        buffer = []

    body = document.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            paragraph = Paragraph(child, document)
            style = (paragraph.style.name or "").lower() if paragraph.style else ""
            text = paragraph.text.strip()
            if not text:
                continue
            if style.startswith("heading"):
                flush()
                current_heading = text
            else:
                buffer.append(text)
        elif child.tag.endswith("}tbl"):
            table = Table(child, document)
            rows = []
            for row in table.rows:
                cells = [clean_text(cell.text) for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                buffer.append("\n".join(rows))
    flush()

    pages: list[Page] = []
    for heading, texts in blocks:
        pages.append(
            Page(
                text="\n\n".join(texts),
                source=path.name,
                path=str(path),
                section=heading,
            )
        )
    if not pages:  # fallback: whole document
        text = clean_text("\n".join(p.text for p in document.paragraphs))
        if text:
            pages.append(Page(text=text, source=path.name, path=str(path)))
    return pages


# --------------------------------------------------------------------------- #
# Plain text / Markdown
# --------------------------------------------------------------------------- #
def load_text(path: Path) -> list[Page]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = clean_text(text)
    if not text:
        return []

    if path.suffix.lower() == ".md":
        # split on top level headings so chunks keep their section context
        parts = re.split(r"\n(?=#{1,2} )", text)
        pages = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            first_line = part.splitlines()[0]
            section = first_line.lstrip("#").strip() if first_line.startswith("#") else None
            pages.append(
                Page(
                    text=part,
                    source=path.name,
                    path=str(path),
                    section=section,
                )
            )
        return pages or [Page(text=text, source=path.name, path=str(path))]

    return [Page(text=text, source=path.name, path=str(path))]


# --------------------------------------------------------------------------- #
# CSV / XLSX - rows become readable "column: value" sentences
# --------------------------------------------------------------------------- #
def _rows_to_pages(
    rows: list[list[str]],
    header: list[str] | None,
    source: str,
    path: Path,
    sheet: str | None,
    rows_per_chunk: int,
    start_row: int = 1,
) -> list[Page]:
    pages: list[Page] = []
    width = max((len(r) for r in rows), default=0)
    header = header or [f"col_{i + 1}" for i in range(width)]

    for offset in range(0, len(rows), rows_per_chunk):
        block = rows[offset : offset + rows_per_chunk]
        lines: list[str] = []
        for row_num, row in enumerate(block, start=start_row + offset):
            pairs = []
            for col_index, value in enumerate(row):
                value = (value or "").strip()
                if not value:
                    continue
                name = header[col_index] if col_index < len(header) else f"col_{col_index + 1}"
                pairs.append(f"{name}: {value}")
            if pairs:
                lines.append(f"Row {row_num} - " + "; ".join(pairs))
        text = "\n".join(lines)
        if text:
            pages.append(
                Page(
                    text=text,
                    source=source,
                    path=str(path),
                    section=sheet,
                    extra={"row_start": start_row + offset, "row_end": start_row + offset + len(block) - 1},
                )
            )
    return pages


def load_csv(path: Path) -> list[Page]:
    raw = path.read_text(encoding="utf-8-sig", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(raw[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(raw), dialect=dialect))
    if not rows:
        return []
    header = rows[0]
    body = rows[1:]
    return _rows_to_pages(
        body, header, path.name, path, None, config.CSV_ROWS_PER_CHUNK, start_row=2
    )


def load_xlsx(path: Path) -> list[Page]:
    from openpyxl import load_workbook

    workbook = load_workbook(str(path), read_only=True, data_only=True)
    pages: list[Page] = []
    for sheet in workbook.worksheets:
        rows = [
            ["" if cell is None else str(cell) for cell in row]
            for row in sheet.iter_rows(values_only=True)
        ]
        rows = [row for row in rows if any(str(c).strip() for c in row)]
        if not rows:
            continue
        header = rows[0]
        body = rows[1:]
        pages.extend(
            _rows_to_pages(
                body,
                header,
                path.name,
                path,
                sheet.title,
                config.CSV_ROWS_PER_CHUNK,
                start_row=2,
            )
        )
    workbook.close()
    return pages


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
_LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_text,
    ".md": load_text,
    ".csv": load_csv,
    ".xlsx": load_xlsx,
    ".xlsm": load_xlsx,
}


def load_file(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file type: {suffix}")
    return loader(path)


def iter_documents() -> list[Path]:
    if not config.DOCS_DIR.exists():
        return []
    return sorted(
        p for p in config.DOCS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_EXTENSIONS
    )
