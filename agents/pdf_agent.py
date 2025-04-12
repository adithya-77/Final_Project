import fitz  # PyMuPDF

class PDFScrapingAgent:
    def scrape(self, uploaded_file):
        text = ""
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in pdf_document:
            text += page.get_text()
        return text
