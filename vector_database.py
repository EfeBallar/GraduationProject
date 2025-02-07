import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv

load_dotenv()
DOC_PATH = os.getenv("DOC_PATH")
V_DB_PATH = os.getenv("V_DB_PATH")

def generate_chunks_from_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Returns:
        List[str]: A list of text chunks extracted from the PDF.
    """
    text = ""
    
    # Open and read the PDF file
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    # Optionally include page metadata for context
                    text += f"\n\n--- Page {i+1} ---\n" + page_text
    except Exception as e:
        print(f"Error reading PDF file {pdf_path}: {e}")
        return []
    
    # Split the extracted text into words
    words = text.split()
    chunks = []
    index = 0

    # Generate overlapping chunks
    while index < len(words):
        chunk_words = words[index: index + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        
        if index + chunk_size >= len(words):
            break
        
        index += chunk_size - chunk_overlap

    return chunks

def process_multiple_pdfs(pdf_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Returns:
        List[dict]: A list of dictionaries. Each dictionary contains:
            - 'pdf': PDF file name.
            - 'chunk': The text chunk.
    """
    chunks_data = []
    for file_name in os.listdir(pdf_dir):
        if file_name.lower().endswith(".pdf"):
            print(f"Processing PDF: {file_name}")
            pdf_path = os.path.join(pdf_dir, file_name)
            chunks = generate_chunks_from_pdf(pdf_path, chunk_size, chunk_overlap)
            for chunk in chunks:
                chunks_data.append({"pdf": file_name, "chunk": chunk})
    return chunks_data

def embed_chunks(chunks_data: list, model_name: str = "all-MiniLM-L6-v2", convert_to_tensor: bool = True) -> list:
    """
    Returns:
        List[dict]: The input list with an added key 'embedding' for each dictionary.
    """
    # Load the pre-trained model
    model = SentenceTransformer(model_name)
    
    # Extract all text chunks for encoding
    texts = [item["chunk"] for item in chunks_data]
    
    # Generate embeddings for all chunks at once
    embeddings = model.encode(texts, convert_to_tensor=convert_to_tensor)
    
    # Attach each embedding to its corresponding dictionary
    for i, item in enumerate(chunks_data):
        item["embedding"] = embeddings[i]
    
    return chunks_data

def save_chunks_to_faiss(chunks_data: list,
                         index_file: str = "faiss_index.idx",
                         metadata_file: str = "metadata.pkl") -> None:
    """
    Save embeddings from chunks_data into a FAISS vector index and store metadata separately.
    """
    embeddings = []
    metadata = []
    
    for item in chunks_data:
        # Ensure the embedding is a NumPy array (convert from tensor if necessary)
        if hasattr(item["embedding"], "cpu"):
            emb = item["embedding"].cpu().numpy()
        else:
            emb = item["embedding"]
        embeddings.append(emb)
        
        # Store metadata (without the embedding to avoid redundancy)
        metadata.append({"pdf": item["pdf"], "chunk": item["chunk"]})
        
    # Stack all embeddings into a 2D NumPy array of type float32
    embeddings_matrix = np.vstack(embeddings).astype("float32")
    
    # Normalize embeddings for cosine similarity (FAISS inner product on normalized vectors mimics cosine similarity)
    faiss.normalize_L2(embeddings_matrix)
    
    embedding_dim = embeddings_matrix.shape[1]
    
    # Create a FAISS index. Here we use IndexFlatIP for inner product similarity.
    index = faiss.IndexFlatIP(embedding_dim)
    
    # Add embeddings to the index
    index.add(embeddings_matrix)
    
    # Save the FAISS index to disk
    faiss.write_index(index, index_file)
    
    # Save the metadata using pickle
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"Saved FAISS index with {index.ntotal} vectors to '{index_file}'.")
    print(f"Saved metadata for {len(metadata)} vectors to '{metadata_file}'.") 

def add_chunks_to_faiss(new_chunks_data: list,
                        index_file: str = "faiss_index.idx",
                        metadata_file: str = "metadata.pkl") -> None:
    """
    Adds new chunk embeddings and metadata to an existing FAISS index and metadata file.
    
    Parameters:
        new_chunks_data (list): List of dictionaries, each containing:
            - 'pdf': PDF file name.
            - 'chunk': Text chunk.
            - 'embedding': The embedding of the chunk (as a PyTorch tensor or NumPy array).
        index_file (str): Filename of the existing FAISS index.
        metadata_file (str): Filename of the metadata pickle file.
    """
    # Load the existing FAISS index
    index = faiss.read_index(index_file)
    
    # Load existing metadata
    try:
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
    except FileNotFoundError:
        print(f"Metadata file '{metadata_file}' not found. Starting with an empty metadata list.")
        metadata = []
    
    new_embeddings = []
    
    # Process new chunks data
    for item in new_chunks_data:
        # Convert embedding to NumPy array if it is a PyTorch tensor
        if hasattr(item["embedding"], "cpu"):
            emb = item["embedding"].cpu().numpy()
        else:
            emb = item["embedding"]
        new_embeddings.append(emb)
        
        # Append the new metadata
        metadata.append({
            "pdf": item["pdf"],
            "chunk": item["chunk"]
        })
    
    # Stack all new embeddings into a matrix (float32)
    new_embeddings_matrix = np.vstack(new_embeddings).astype("float32")
    
    # Normalize embeddings (if your index uses normalized vectors for cosine similarity)
    faiss.normalize_L2(new_embeddings_matrix)
    
    # Add new embeddings to the FAISS index
    index.add(new_embeddings_matrix)
    
    # Save the updated FAISS index and metadata
    faiss.write_index(index, index_file)
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"Added {len(new_chunks_data)} new vectors. Total vectors in index: {index.ntotal}")

def delete_chunks_from_file(file_name: str,
                            index_file: str = "faiss_index.idx",
                            metadata_file: str = "metadata.pkl") -> None:
    """
    Delete all vector chunks corresponding to a given file from the FAISS vector database.
    
    This function assumes that:
      - The FAISS index was created as an IndexIDMap, so that each vector has a unique ID.
      - The associated metadata (stored in a pickle file) is a list of dictionaries, and each
        dictionary includes:
            - 'pdf': the file name from which the chunk came.
            - 'chunk': the text of the chunk.
            - 'id': the unique identifier corresponding to the vector in the FAISS index.
    
    Parameters:
        file_name (str): The name of the file whose chunks should be deleted.
        index_file (str): Path to the FAISS index file.
        metadata_file (str): Path to the metadata pickle file.
    """
    # Load the FAISS index
    index = faiss.read_index(index_file)
    
    # Load the existing metadata
    try:
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
    except FileNotFoundError:
        print(f"Metadata file '{metadata_file}' not found.")
        return

    # Identify IDs for chunks that belong to the specified file.
    ids_to_delete = []
    updated_metadata = []
    for item in metadata:
        # We assume each metadata item has an 'id' field (assigned when the vector was added)
        if item.get("pdf") == file_name:
            if "id" in item:
                ids_to_delete.append(item["id"])
            else:
                print("Error: Metadata item missing 'id'. Cannot delete without vector IDs.")
                return
        else:
            updated_metadata.append(item)
    
    if not ids_to_delete:
        print(f"No chunks found for file '{file_name}'.")
        return

    # Convert the list of IDs to a NumPy array of type int64
    ids_to_delete = np.array(ids_to_delete, dtype=np.int64)
    
    # Remove the vectors corresponding to these IDs from the FAISS index.
    # Note: This works only if the index is wrapped in an IndexIDMap.
    index.remove_ids(ids_to_delete)
    
    # Save the updated index and metadata back to disk
    faiss.write_index(index, index_file)
    with open(metadata_file, "wb") as f:
        pickle.dump(updated_metadata, f)
    
    print(f"Deleted {len(ids_to_delete)} chunks from file '{file_name}'.")

if __name__ == "__main__":
    course_codes = ['CS307']
    for course_code in course_codes:
        pdf_directory = DOC_PATH + "\\" + course_code
        # Process PDFs and generate text chunks
        chunks_data = process_multiple_pdfs(pdf_directory, chunk_size=1000, chunk_overlap=200)
        print(f"Extracted {len(chunks_data)} chunks from PDF files.")
    
        # Generate embeddings for each chunk using Sentence Transformers
        chunks_data = embed_chunks(chunks_data, model_name="all-MiniLM-L6-v2", convert_to_tensor=True)
        print("Generated embeddings for all chunks.")
    
        # Save the chunks and their embeddings to a FAISS vector database (with metadata)
        save_chunks_to_faiss(chunks_data, index_file= V_DB_PATH + "\\" + course_code+"_faiss_index.idx", metadata_file=V_DB_PATH + "\\" + course_code+"_metadata.pkl")
        print(f"Processed the documents for {course_code}.")