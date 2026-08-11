from pathlib import Path

from pypdf import PdfReader

class PDFLoadError(Exception):
    """Custom exception for PDF loading errors."""
    pass

def load_pdf(file_path: Path) -> str:
    try:
        reader = PdfReader(file_path)
        
        pages = []
        
        for pagenum, page in enumerate(reader.pages,start = 1):
            text = page.extract_text() or ""
            
            pages.append(
                {
                    "page_number": pagenum,
                    "text": text
                }
            )
        return pages
    except Exception as e:
        raise PDFLoadError(f"Unable to read PDF file: {e}")