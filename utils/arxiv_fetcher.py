import asyncio
import arxiv
import pypdf
import ssl
from typing import Tuple, List
from pathlib import Path

from core.config import settings
from core.logging import get_logger

logger = get_logger()

async def fetch_arxiv_paper(arxiv_id: str) -> Tuple[str, str, List[str]]:
    """Download and extract text from arXiv paper"""
    text, paper_title, paper_authors = await asyncio.to_thread(_fetch_sync, arxiv_id)
    return text, paper_title, paper_authors


def _fetch_sync(arxiv_id: str) -> Tuple[str, str, List[str]]:
    """Download and extract text from arXiv paper"""
    # Create SSL context that doesn't verify certificates
    # This is needed on some Windows systems where certificate verification fails
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # Create client with custom SSL context
    import urllib.request
    import urllib.error
    
    # Monkey patch urllib to use our SSL context
    original_urlopen = urllib.request.urlopen
    def urlopen_with_context(url, *args, **kwargs):
        if 'context' not in kwargs:
            kwargs['context'] = ssl_context
        return original_urlopen(url, *args, **kwargs)
    
    urllib.request.urlopen = urlopen_with_context
    
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(client.results(search))

        data_dir = Path(settings.ARXIV_DIR)
        data_dir.mkdir(exist_ok=True, parents=True)
        
        # Download PDF - arxiv library names it with version and title
        # e.g., "1706.03762v7.Attention_Is_All_You_Need.pdf"
        paper.download_pdf(dirpath=str(data_dir))
        
        # Find the actual downloaded PDF file (arxiv adds version and title to filename)
        # Look for any PDF file that starts with the arxiv_id
        pdf_files = list(data_dir.glob(f"{arxiv_id}*.pdf"))
        
        if not pdf_files:
            raise FileNotFoundError(f"Downloaded PDF not found for arxiv_id: {arxiv_id}")
        
        # Use the first matching file (should only be one)
        pdf_path = pdf_files[0]
        logger.info(f"Found PDF: {pdf_path.name}")
        
        # Extract text
        with open(str(pdf_path), 'rb') as file:
            reader = pypdf.PdfReader(file)
            text = " ".join([page.extract_text() for page in reader.pages])
        
        # Convert Author objects to strings
        # arxiv library returns Author objects with .name attribute
        author_names = [author.name for author in paper.authors]
        
        logger.info(f"Extracted {len(text)} characters from {pdf_path.name}")
        logger.info(f"Authors: {', '.join(author_names[:3])}...")
        
        return text, paper.title, author_names
    
    finally:
        # Restore original urlopen
        urllib.request.urlopen = original_urlopen