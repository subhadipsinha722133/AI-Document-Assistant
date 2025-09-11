import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import tempfile
import os
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import cv2
import numpy as np
import io

def preprocess_image_for_ocr(image):
    """
    Enhanced image preprocessing for better OCR results
    """
    try:
        # Convert PIL Image to OpenCV format
        img = np.array(image)
        
        # Convert to grayscale if not already
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply various preprocessing techniques
        # 1. Noise reduction
        img = cv2.medianBlur(img, 3)
        
        # 2. Thresholding
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Morphological operations to clean up text
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        # 4. Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
        
        return Image.fromarray(img)
    except Exception as e:
        print(f"Image preprocessing failed: {e}")
        return image

def extract_text_with_enhanced_ocr(pdf_path, dpi=300):
    """
    Enhanced OCR extraction with multiple strategies
    """
    text = ""
    
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=dpi, thread_count=4)
        
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)} with OCR...")
            
            # Try multiple preprocessing strategies
            strategies = [
                lambda img: img,  # Original image
                lambda img: preprocess_image_for_ocr(img),  # Standard preprocessing
            ]
            
            page_text = ""
            for strategy_idx, strategy in enumerate(strategies):
                try:
                    processed_image = strategy(image)
                    
                    # Try different OCR configurations
                    ocr_configs = [
                        '--oem 3 --psm 6',  # Default
                        '--oem 3 --psm 4',  # Assume single column of text
                        '--oem 3 --psm 8',  # Single word
                        '--oem 1 --psm 6',  # Legacy engine
                    ]
                    
                    for config in ocr_configs:
                        try:
                            strategy_text = pytesseract.image_to_string(
                                processed_image, 
                                lang='eng', 
                                config=config
                            )
                            
                            if strategy_text and len(strategy_text.strip()) > len(page_text.strip()):
                                page_text = strategy_text
                        except:
                            continue
                            
                except Exception as e:
                    print(f"OCR strategy {strategy_idx} failed: {e}")
                    continue
            
            if page_text.strip():
                text += f"--- Page {i+1} ---\n{page_text}\n\n"
            else:
                print(f"Warning: No text extracted from page {i+1}")
    
    except Exception as e:
        print(f"Enhanced OCR extraction failed: {e}")
        # Fallback to simple OCR
        try:
            images = convert_from_path(pdf_path, dpi=200)
            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image, lang='eng')
                if page_text.strip():
                    text += f"--- Page {i+1} ---\n{page_text}\n\n"
        except:
            pass
    
    return text

def try_direct_text_extraction(pdf_path):
    """
    Try all direct text extraction methods
    """
    text = ""
    methods = []
    
    # Method 1: PyMuPDF (most reliable)
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                page_text = page.get_text()
                if page_text and page_text.strip():
                    text += page_text + "\n"
                    methods.append("PyMuPDF")
    except:
        pass
    
    # Method 2: pdfplumber
    if not text.strip():
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += page_text + "\n"
                        methods.append("pdfplumber")
        except:
            pass
    
    # Method 3: PyPDF2
    if not text.strip():
        try:
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += page_text + "\n"
                        methods.append("PyPDF2")
        except:
            pass
    
    return text, methods

def extract_text_from_pdf(pdf_file):
    """
    Comprehensive text extraction with multiple fallbacks
    """
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name
    
    text = ""
    methods_used = []
    
    try:
        # First try direct text extraction
        text, direct_methods = try_direct_text_extraction(tmp_path)
        methods_used.extend(direct_methods)
        
        # If direct methods failed, try OCR
        if not text.strip():
            print("Direct extraction failed, trying OCR...")
            text = extract_text_with_enhanced_ocr(tmp_path, dpi=300)
            if text.strip():
                methods_used.append("OCR")
        
        # If still no text, try extreme measures
        if not text.strip():
            print("All methods failed, trying extreme measures...")
            
            # Try with higher DPI
            text = extract_text_with_enhanced_ocr(tmp_path, dpi=400)
            if text.strip():
                methods_used.append("High-DPI OCR")
            
            # Try different languages
            if not text.strip():
                try:
                    images = convert_from_path(tmp_path, dpi=300)
                    for i, image in enumerate(images):
                        try:
                            page_text = pytesseract.image_to_string(image, lang='eng+fra+deu+spa')
                            if page_text.strip():
                                text += f"--- Page {i+1} ---\n{page_text}\n\n"
                                methods_used.append("Multi-language OCR")
                        except:
                            pass
                except:
                    pass
    
    except Exception as e:
        print(f"Comprehensive extraction failed: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    print(f"Extraction methods used: {methods_used}")
    print(f"Extracted text length: {len(text)}")
    
    return text

def chunk_text(text, chunk_size=800, chunk_overlap=150):
    """
    Split text into chunks for processing with better handling
    """
    # Check if we have valid text to process
    if not text or not text.strip():
        raise ValueError("No text extracted from PDF using any method.")
    
    # Clean up text
    text = text.strip()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Create documents for LangChain
    documents = [Document(page_content=text)]
    chunks = text_splitter.split_documents(documents)
    
    return chunks