import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

# Directory to save PDFs
SAVE_DIR = "downloaded_pdfs"
os.makedirs(SAVE_DIR, exist_ok=True)

# Website URL
BASE_URL = "https://semico.ir/wp-content/uploads"

# Set to track visited URLs to avoid revisiting them
visited_urls = set()

def crawl_and_download_pdfs(base_url, save_dir):
    try:
        # If the URL was already visited, skip it
        if base_url in visited_urls:
            return
        
        # Mark the URL as visited
        visited_urls.add(base_url)
        
        # Fetch the website content
        response = requests.get(base_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all links (anchors)
        links = soup.find_all("a", href=True)
        
        # Loop through links and find PDFs or subdirectories
        for link in links:
            href = link["href"]
            print(f"Found link: {href}")  # Debugging line
            full_url = urljoin(base_url, href)  # Handle relative URLs
            
            # If the link is a directory (i.e., it ends with '/'), crawl it recursively
            if full_url.endswith('/'):
                crawl_and_download_pdfs(full_url, save_dir)
            # If the link is a PDF, download it
            elif href.endswith(".pdf"):
                download_pdf(full_url, save_dir)

    except Exception as e:
        print(f"Error during crawling: {e}")

def download_pdf(pdf_url, save_dir):
    try:
        # Get the PDF content
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()  # Ensure valid response

        # Extract filename from URL
        filename = os.path.basename(pdf_url)
        save_path = os.path.join(save_dir, filename)

        # Save the PDF locally
        with open(save_path, "wb") as pdf_file:
            for chunk in response.iter_content(chunk_size=1024):
                pdf_file.write(chunk)
        
        print(f"Downloaded: {filename}")

    except Exception as e:
        print(f"Error downloading {pdf_url}: {e}")

if __name__ == "__main__":
    crawl_and_download_pdfs(BASE_URL, SAVE_DIR)
