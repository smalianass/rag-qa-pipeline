from docling.document_converter import DocumentConverter

pdf_path = "data/docker_cheatsheet.pdf"
markdown_path = "data/docker_cheatsheet.md"

converter = DocumentConverter()

result = converter.convert(pdf_path)

markdown_content = result.document.export_to_markdown()

with open(markdown_path, "w", encoding="utf-8") as f:
    f.write(markdown_content)

print(f"Markdown créé : {markdown_path}")