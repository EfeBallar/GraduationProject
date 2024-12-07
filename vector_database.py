from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil
import time

MODEL="llama3.1:8b"
CHROMA_PATH = "chroma"
DATA_PATH = "Data"


def create_vector_database(term, course_code):
    start_time = time.time()
    generate_data_store(term, course_code)
    end_time = time.time()
    print(f"Database creation time: {end_time - start_time:.2f} seconds")


def generate_data_store(term, course_code):
    documents = load_documents(term, course_code)
    chunks = split_text(documents)
    save_to_chroma(chunks, term, course_code)


def load_documents(term, course_code):
    loader = DirectoryLoader(f"{DATA_PATH}/{term}/{course_code}") #glob can be specified
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,#10%
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["page_number"] = i + 1  # Add page number metadata
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks


def save_to_chroma(chunks: list[Document], term, course_code):
    # Clear out the database first.
    course_chroma_path = f"{CHROMA_PATH}/{term}/{course_code}"
    if os.path.exists(course_chroma_path):
        shutil.rmtree(course_chroma_path)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OllamaEmbeddings(model=MODEL), persist_directory=course_chroma_path
    )
    print(f"Saved {len(chunks)} chunks to {course_chroma_path}.")

if __name__=="__main__":
    term = input("Enter academic term: ")
    course_code = input("Enter course code: ")
    create_vector_database(term, course_code)
