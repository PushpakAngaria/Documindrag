import os
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import io

class OCRProcessor:
    """Handles the extraction of text from images or scanned PDFs using Tesseract OCR."""
    
    def __init__(self):
        # Configure Tesseract path if necessary (especially on Windows)
        if os.name == 'nt':
            common_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Users\\' + os.getlogin() + r'\AppData\Local\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
            ]
            for path in common_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"[OCR] Found Tesseract at: {path}")
                    break

    def extract_text_from_image(self, image_path_or_obj):
        """Extracts text from a single image file."""
        try:
            image = Image.open(image_path_or_obj)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"Error processing image: {e}")
            return ""

    def extract_text_from_scanned_pdf(self, pdf_path):
        """
        Converts a scanned PDF to images and extracts text.
        Requires poppler to be installed and in PATH.
        """
        text = ""
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path)
            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image)
                text += f"--- Page {i+1} ---\n{page_text}\n"
        except Exception as e:
            print(f"Error processing scanned PDF: {e}")
            raise e
            
        return text

    def process_document(self, file_path_or_obj, is_pdf=False):
        """
        Main method to process an image or scanned PDF.
        """
        if is_pdf:
            # Note: poppler is required for this to work
            return self.extract_text_from_scanned_pdf(file_path_or_obj)
        else:
            return self.extract_text_from_image(file_path_or_obj)
