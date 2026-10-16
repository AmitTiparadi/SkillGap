from pathlib import Path
import unittest

from skillgap.parsers import DocumentError, extract_text, resume_lines, split_jobs


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class ParserTests(unittest.TestCase):
    def test_text_preserves_citation_wording(self):
        text = extract_text((EXAMPLES / "resume.txt").read_bytes(), "resume.txt")
        self.assertIn("Built a Flask REST API for a student library with SQLite and unit tests.", resume_lines(text))

    def test_pdf_sample(self):
        text = extract_text((EXAMPLES / "resume.pdf").read_bytes(), "resume.pdf")
        self.assertIn("Alex Student", text)
        self.assertIn("SQLite and unit tests.", text)

    def test_invalid_inputs(self):
        for data, name in [(b"", "resume.txt"), (b"\xff", "resume.txt"),
                           (b"text", "resume.docx"), (b"broken", "resume.pdf")]:
            with self.subTest(name=name, data=data), self.assertRaises(DocumentError):
                extract_text(data, name)

    def test_blank_pdf_requires_ocr(self):
        from io import BytesIO
        from reportlab.pdfgen import canvas
        stream = BytesIO()
        document = canvas.Canvas(stream)
        document.showPage()
        document.save()
        with self.assertRaisesRegex(DocumentError, "OCR"):
            extract_text(stream.getvalue(), "scan.pdf")

    def test_jobs_are_explicitly_split_and_deduplicated(self):
        self.assertEqual(split_jobs("SQL\n---\nPython\n---\nSQL"), ["SQL", "Python"])


if __name__ == "__main__":
    unittest.main()
