from docling.document_converter import DocumentConverter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
pdf_path = DATA_DIR / "docker_cheatsheet.pdf"
markdown_path = DATA_DIR / "docker_cheatsheet.md"

DATA_DIR.mkdir(parents=True, exist_ok=True)
converter = DocumentConverter()

result = converter.convert(pdf_path)

markdown_content = result.document.export_to_markdown()

with markdown_path.open("w", encoding="utf-8") as f:
    f.write(markdown_content)

print(f"Markdown créé : {markdown_path}")