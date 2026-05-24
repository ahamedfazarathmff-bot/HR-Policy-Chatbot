import os
from pypdf import PdfReader

def read_pdf(pdf_path):          # ✅ padf → pdf
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")
    
    reader = PdfReader(pdf_path)  # ✅ colon நீக்கவும்
    pages = [page.extract_text() for page in reader.pages]  # ✅ pages → page
    return pages