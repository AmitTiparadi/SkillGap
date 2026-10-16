"""Extract document text without uploading it or rewriting citation lines."""

from io import BytesIO
from pathlib import Path


class DocumentError(ValueError):
    """An actionable input error the UI can display safely."""


def extract_text(data: bytes, filename: str) -> str:
    """Read an uploaded PDF or UTF-8 text file entirely in memory.

    pdfplumber handles text PDFs, not OCR. Do not silently treat a scanned
    document as an empty resume: that would produce misleading missing grades.
    """
    suffix = Path(filename).suffix.lower()
    if not data:
        raise DocumentError("The resume file is empty.")
    if suffix == ".txt":
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise DocumentError("Save the text resume with UTF-8 encoding.") from error
    elif suffix == ".pdf":
        import pdfplumber

        try:
            with pdfplumber.open(BytesIO(data)) as document:
                pages = [page.extract_text() or "" for page in document.pages]
            if any(not page.strip() for page in pages):
                raise DocumentError("A PDF page has no extractable text. Use OCR or upload a text version.")
            text = "\n\n".join(pages)
        except DocumentError:
            raise
        except Exception as error:
            raise DocumentError("Could not read the PDF. Check that it is valid and not password protected.") from error
    else:
        raise DocumentError("Upload a .pdf or .txt resume.")
    # Normalize only line endings. Keep punctuation, bullet marks, and wording.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise DocumentError("The resume contains no readable text.")
    if "\x00" in text:
        raise DocumentError("The resume contains binary data; upload readable UTF-8 text.")
    return text


def resume_lines(text: str) -> list[str]:
    """Short evidence units copied verbatim from the extracted resume.

    Outer whitespace is removed, but long lines are not truncated: citations
    should retain qualifiers such as 'learning' or 'no production experience'.
    """
    return list(dict.fromkeys(line.strip() for line in text.splitlines() if line.strip()))


def split_jobs(text: str) -> list[str]:
    """Separate pasted jobs with a line containing exactly three hyphens."""
    jobs: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip() == "---":
            if "\n".join(current).strip():
                jobs.append("\n".join(current).strip())
            current = []
        else:
            current.append(line)
    if "\n".join(current).strip():
        jobs.append("\n".join(current).strip())
    # The same posting pasted twice must not inflate gap frequency.
    return list(dict.fromkeys(jobs))
