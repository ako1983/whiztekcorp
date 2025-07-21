"""
pdf_loader.py
-------------
Loads a PDF file and extracts text content and metadata.
Uses LangChain's PyPDFLoader with enhanced error handling and metadata extraction.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Dict, Optional, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_pdf(path: str) -> List[Dict[str, Any]]:
    """
    Load a PDF and return a list of page-wise dictionaries with text and metadata.
    Each dictionary contains:
        - page_number: Page number (1-indexed)
        - text_content: Extracted text from the page
        - metadata: Additional information about the page

    :param path: Path to the PDF file
    :return: List of dictionaries with page text and metadata
    :raises FileNotFoundError: If the PDF file does not exist
    :raises ValueError: If the file is not a valid PDF
    """
    # Validate file exists
    if not os.path.exists(path):
        logger.error(f"PDF file not found: {path}")
        raise FileNotFoundError(f"PDF file not found: {path}")
    
    # Validate file extension
    if not path.lower().endswith('.pdf'):
        logger.warning(f"File may not be a PDF: {path}")
    
    try:
        logger.info(f"Loading PDF: {path}")
        loader = PyPDFLoader(path)
        docs = loader.load()  # List of Document objects, one per page
        
        pages = []
        for i, doc in enumerate(docs):
            # Extract metadata from LangChain document if available
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            
            pages.append({
                "page_number": i + 1,
                "text_content": doc.page_content.strip(),
                "metadata": metadata
            })
        
        logger.info(f"Successfully loaded {len(pages)} pages from PDF")
        return pages
        
    except Exception as e:
        logger.error(f"Error loading PDF {path}: {str(e)}")
        raise ValueError(f"Failed to load PDF: {str(e)}")

def get_pdf_info(path: str) -> Dict[str, Any]:
    """
    Get basic information about a PDF file without extracting all text.
    
    :param path: Path to the PDF file
    :return: Dictionary with basic PDF info (page count, title, etc.)
    :raises FileNotFoundError: If the PDF file does not exist
    """
    if not os.path.exists(path):
        logger.error(f"PDF file not found: {path}")
        raise FileNotFoundError(f"PDF file not found: {path}")
    
    try:
        logger.info(f"Getting PDF info for: {path}")
        loader = PyPDFLoader(path)
        docs = loader.load()
        
        # Get basic file information
        file_info = {
            "filename": os.path.basename(path),
            "page_count": len(docs),
            "file_size_bytes": os.path.getsize(path),
            "last_modified": os.path.getmtime(path)
        }
        
        # Try to extract document metadata if available in the first page
        if len(docs) > 0 and hasattr(docs[0], 'metadata'):
            metadata = docs[0].metadata
            if metadata:
                file_info["metadata"] = metadata
        
        return file_info
        
    except Exception as e:
        logger.error(f"Error getting PDF info for {path}: {str(e)}")
        raise ValueError(f"Failed to get PDF info: {str(e)}")

# Example usage
if __name__ == "__main__":
    import sys
    import json
    
    # Allow specifying a PDF path as a command-line argument
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "example.pdf"
    
    try:
        # Get basic PDF info
        info = get_pdf_info(pdf_path)
        print("\nPDF Information:")
        print(json.dumps(info, indent=2))
        
        # Load the first page as a demo
        output = load_pdf(pdf_path)
        print("\nSample Page Content:")
        print(json.dumps(output[:1], indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
