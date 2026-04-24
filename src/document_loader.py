import os
import PyPDF2

class DocumentLoader:
    """Handles the extraction of text from standard, machine-readable PDF files."""
    
    def __init__(self):
        pass

    def extract_text_from_pdf(self, file_path_or_file_obj):
        """
        Extracts text from a given PDF file.
        Returns a string of extracted text.
        """
        text = ""
        try:
            # If it's a path string
            if isinstance(file_path_or_file_obj, str):
                with open(file_path_or_file_obj, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
            # If it's a file-like object (e.g. from Streamlit)
            else:
                reader = PyPDF2.PdfReader(file_path_or_file_obj)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")
            raise e
            
        return text

    def process_document(self, file_path_or_file_obj):
        """Main method to process a standard PDF document."""
        return self.extract_text_from_pdf(file_path_or_file_obj)
