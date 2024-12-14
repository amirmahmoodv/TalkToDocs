from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import easyocr
from pdf2image import convert_from_path
import os
import numpy as np
import hashlib
import shutil
import time


CACHE_DIR = "assets\\ocr_cache"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
    
    
def get_file_hash(file_path):
    """Generate a hash of the file content for caching purposes."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def extract_text_from_image_pdf(pdf_path):
    # Generate a unique hash for the PDF based on its content
    pdf_hash = get_file_hash(pdf_path)
    cached_file = os.path.join(CACHE_DIR, f"{pdf_hash}.txt")
    
    # Check if OCR result is cached
    if os.path.exists(cached_file):
        print(f"OCR result for {pdf_path} already cached. Loading...")
        with open(cached_file, "r", encoding="utf-8") as f:
            return f.read()

    print("Attempting OCR on image-based PDF...")
    reader = easyocr.Reader(['fa', 'en'], gpu=True)  # Switch to CPU if GPU issues occur

    try:
        images = convert_from_path(pdf_path)
        extracted_text = ""
        for page_number, image in enumerate(images, start=1):
            print(f"Processing page {page_number} with OCR...")
            
            # Convert Pillow image to numpy array
            image_np = np.array(image)
            
            # Perform OCR on the numpy array
            text = reader.readtext(image_np, detail=0, paragraph=True)
            extracted_text += f"\n--- Page {page_number} ---\n" + "\n".join(text)
        
        # Cache the OCR result for future use
        with open(cached_file, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        
        return extracted_text
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return ""

# Check if PDF is image-based or text-based
def extract_text_from_pdf(pdf_path):
    pdf_hash = get_file_hash(pdf_path)
    cached_file = os.path.join(CACHE_DIR, f"{pdf_hash}.txt")
    try:
        
        pdfreader = PdfReader(pdf_path)
        extracted_text = ""
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                extracted_text += f"\n--- Page {page} ---\n" + content
                
        if extracted_text.strip():  # If text-based content exists
            with open(cached_file, "w", encoding="utf-8") as f:
                f.write(extracted_text)
            return extracted_text
        else:
            print("No text found in PDF. Processing as an image-based PDF...")
            return extract_text_from_image_pdf(pdf_path)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return extract_text_from_image_pdf(pdf_path)






cwd = os.getcwd()
os.environ["OPENAI_API_KEY"]



# pdfreader = PdfReader(f'{pdf_dir}Autumn_Budget_2024__web_accessible_.pdf')
pdf_dir = os.path.join(cwd, "assets", "downloaded_pdfs")
processed_dir = os.path.join(cwd, "assets", "processed")

# Ensure the processed directory exists
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(pdf_dir, exist_ok=True)


# Path to PDF

# if not os.path.exists(pdf_path):
#     print(f"File not found: {pdf_path}")
    



# Extract text from PDF
# raw_text = extract_text_from_pdf(pdf_path)
# dir = os.path.join(cwd,'assets','s.txt')
# with open(dir, 'r') as file:
#     content = file.read()
    

# raw_text = content


def process_pdfs(pdf_dir):
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            print(f"Processing: {pdf_path}")

            # Extract text from PDF
            raw_text = extract_text_from_pdf(pdf_path)

            if raw_text.strip():
                print(f"Text extracted from {filename}:\n{raw_text[:500]}...")  # Print the first 500 chars of extracted text
            else:
                print(f"No text found in {filename}, possibly an image-based PDF. OCR applied.")
            
            # Move processed PDF to the 'processed' directory
            processed_pdf_path = os.path.join(processed_dir, filename)
            shutil.move(pdf_path, processed_pdf_path)
            print(f"Moved {filename} to processed directory.")

            # Optional: Add a delay to avoid overwhelming the system
            time.sleep(1)  # Sleep for 1 second between each file to manage load

# Start processing PDFs

process_pdfs(pdf_dir)


# # Step 2: Split Text into Manageable Chunks
# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=800,
#     chunk_overlap=200,
#     length_function=len,
# )
# texts = text_splitter.split_text(raw_text)


# # Step 3: Create Embeddings and Vector Store

# embeddings = OpenAIEmbeddings()
# document_search = FAISS.from_texts(texts, embeddings)

# # Step 4: Create and Run QA Chain
# chain = load_qa_chain(OpenAI(), chain_type="stuff")

# query = "این سند درباره چی هستش؟"
# docs = document_search.similarity_search(query)
# response = chain.run(input_documents=docs, question=query)

# print("Response:", response)