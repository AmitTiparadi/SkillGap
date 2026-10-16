"""Rebuild the synthetic parser fixture: python -m skillgap.examples.create_sample_pdf."""
from pathlib import Path
from reportlab.pdfgen import canvas


def main():
    folder = Path(__file__).resolve().parent
    document = canvas.Canvas(str(folder / "resume.pdf"), pagesize=(612, 792))
    document.setTitle("SkillGap synthetic resume fixture")
    document.setFont("Helvetica", 11)
    y = 740
    for line in (folder / "resume.txt").read_text(encoding="utf-8").splitlines():
        document.drawString(40, y, line)
        y -= 24
    document.save()


if __name__ == "__main__":
    main()
