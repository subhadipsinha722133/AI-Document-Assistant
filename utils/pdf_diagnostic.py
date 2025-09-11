import fitz  # PyMuPDF
import tempfile
import os
from PIL import Image
import io

def analyze_pdf(pdf_file):
    """
    Analyze PDF structure to understand why text extraction is failing
    """
    results = {
        "has_text": False,
        "has_images": False,
        "page_count": 0,
        "file_size": 0,
        "is_encrypted": False,
        "analysis": ""
    }
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name
        results["file_size"] = os.path.getsize(tmp_path)
    
    try:
        with fitz.open(tmp_path) as doc:
            results["page_count"] = len(doc)
            results["is_encrypted"] = doc.is_encrypted
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Check for text
                text = page.get_text()
                if text and text.strip():
                    results["has_text"] = True
                
                # Check for images
                image_list = page.get_images()
                if image_list:
                    results["has_images"] = True
                
                # Extract more detailed info
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if block["type"] == 0:  # text block
                        results["has_text"] = True
                    elif block["type"] == 1:  # image block
                        results["has_images"] = True
            
            # Generate analysis
            if results["is_encrypted"]:
                results["analysis"] = "PDF is encrypted/protected"
            elif results["has_text"]:
                results["analysis"] = "PDF contains text but extraction failed"
            elif results["has_images"] and not results["has_text"]:
                results["analysis"] = "PDF appears to be scanned images only"
            else:
                results["analysis"] = "PDF structure is unusual - may be empty or malformed"
                
    except Exception as e:
        results["analysis"] = f"Error analyzing PDF: {str(e)}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    return results

def extract_images_from_pdf(pdf_file, max_pages=3):
    """
    Extract sample images from PDF for manual inspection
    """
    images = []
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name
    
    try:
        with fitz.open(tmp_path) as doc:
            for page_num in range(min(len(doc), max_pages)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                if image_list:
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image = Image.open(io.BytesIO(image_bytes))
                        images.append((page_num, img_index, image))
                        break  # Just get first image per page
    except Exception as e:
        print(f"Error extracting images: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    return images