from src.file_parsing.parse_docx_with_sections import parse_docx_to_sections
from src.file_parsing.parse_pdf_with_sections import parse_pdf_to_sections
from src.file_parsing.parse_xlsx_with_sections import parse_xlsx_to_sections


def parse_file(file_path: str, ext: str()):
    ext = ext.lower()
    if ext == '.docx' or ext == '.doc':
        return parse_docx_to_sections(file_path)
    elif ext == '.pdf':
        return parse_pdf_to_sections(file_path)
    elif ext == '.xlsx' or ext == '.xls':
        return parse_xlsx_to_sections(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
