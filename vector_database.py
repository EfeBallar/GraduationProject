from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from tqdm import tqdm
import shutil
import time
import os

load_dotenv()
MODEL=os.getenv("MODEL")
CHROMA_PATH = os.getenv("CHROMA_PATH")
DATA_PATH = "data"


def create_vector_database(term, course_code):
    start_time = time.time()
    generate_data_store(term, course_code)
    end_time = time.time()
    print(f"Database creation time: {end_time - start_time:.2f} seconds")


def generate_data_store(term, course_code):
    documents = load_documents(term, course_code)
    if not documents:
        print(f"No documents found for {course_code}.")
        return
    chunks = split_text(documents)
    save_to_chroma(chunks, term, course_code)


def load_documents(term, course_code):
    loader = DirectoryLoader(
                f"{DATA_PATH}/{term}/{course_code}",
                loader_cls=PyPDFLoader,
                show_progress=True,
                use_multithreading=True
            ) #glob can be specified
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,  # 20% overlap
        length_function=len,
        add_start_index=True,
    )
    
    # Process each document to extract page numbers
    chunks = []
    for doc in documents:
        # Extract page numbers if available in metadata
        page_content = doc.page_content
        metadata = doc.metadata
        
        # Split the individual document
        doc_chunks = text_splitter.split_documents([doc])
        
        # Add page number to metadata if available
        for chunk in doc_chunks:
            if 'page' in metadata:
                chunk.metadata['page'] = metadata['page'] + 1
            chunk.metadata["source"] = metadata.get("source", f"{term}/{course_code}")
            chunks.append(chunk)

    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def save_to_chroma(chunks: list[Document], term, course_code):
    # Define the path and clear out the existing database.
    course_chroma_path = f"{CHROMA_PATH}/{term}/{course_code}"
    if os.path.exists(course_chroma_path):
        shutil.rmtree(course_chroma_path)

    db = Chroma(
        embedding_function=OllamaEmbeddings(model=MODEL),
        persist_directory=course_chroma_path,
        collection_metadata={"hnsw:space": "cosine"}
    )
    # Add each chunk individually (or in batches) with a progress bar.
    for chunk in tqdm(chunks, desc="Saving chunks to Chroma"):
        db.add_documents([chunk])
    
    print(f"Saved {len(chunks)} chunks to {course_chroma_path}.")

if __name__=="__main__":  
    term = 'F24-25'
    # course_codes = ['CS302','CS404','CS305','CS307']
    course_codes = ['CS307']
    for course_code in course_codes:
        create_vector_database(term, course_code)
        print(f"Processed the documents for {course_code}.")