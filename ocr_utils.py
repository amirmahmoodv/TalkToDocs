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


def extract_text_from_image_pdf(pdf_path):
    print("Attempting OCR on image-based PDF...")
    reader = easyocr.Reader(['fa', 'en'], gpu=False)  # Switch to CPU if GPU issues occur

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
        return extracted_text
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return ""


# Check if PDF is image-based or text-based
def extract_text_from_pdf(pdf_path):
    try:
        pdfreader = PdfReader(pdf_path)
        raw_text = ""
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                raw_text += content
        if raw_text.strip():  # If text-based content exists
            return raw_text
        else:
            print("No text found in PDF. Processing as an image-based PDF...")
            return extract_text_from_image_pdf(pdf_path)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return extract_text_from_image_pdf(pdf_path)






cwd = os.getcwd()
os.environ["OPENAI_API_KEY"]



# pdfreader = PdfReader(f'{pdf_dir}Autumn_Budget_2024__web_accessible_.pdf')
pdf_dir = os.path.join(cwd, "assets", "pdf")
pdf_path = os.path.join(pdf_dir, "1402-06-31.pdf")

# Path to PDF

if not os.path.exists(pdf_path):
    print(f"File not found: {pdf_path}")
    



# Extract text from PDF
raw_text = extract_text_from_pdf(pdf_path)
print(raw_text)

# # Step 2: Split Text into Manageable Chunks
# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=800,
#     chunk_overlap=200,
#     length_function=len,
# )
# texts = text_splitter.split_text(raw_text)
# print(texts)

# # Step 3: Create Embeddings and Vector Store

# embeddings = OpenAIEmbeddings()
# document_search = FAISS.from_texts(texts, embeddings)

# # Step 4: Create and Run QA Chain
# chain = load_qa_chain(OpenAI(), chain_type="stuff")

# query = "What is the new no-budget fiscal rule?"
# docs = document_search.similarity_search(query)
# response = chain.run(input_documents=docs, question=query)

# print("Response:", response)