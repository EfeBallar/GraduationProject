from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from datetime import datetime, timezone
from langchain_chroma import Chroma
from flask import request, jsonify
from dotenv import load_dotenv
from bson import ObjectId
import torch
import time
import faiss
import pickle
import os

load_dotenv()
LLM_MODEL=os.getenv("LLM_MODEL")
V_DB_PATH = os.getenv("V_DB_PATH")
llm_model = ChatOllama(model=LLM_MODEL, temperature = 0)

CONTENT_PROMPT = """
    You are provided with the following context extracted from a document.
    Please answer the question using only the information provided in the context.
    Do not include any additional knowledge or assumptions beyond what is given.
    If the answer cannot be determined from the context, explicitly state that the information is not available.

    Context:
    {context}

    Question:
    {question}

     Answer:
"""

def query(course_db):
    course = request.json.get('course')
    question = request.json.get('question')
    user_id = request.json.get('user_id')
    chat_id = request.json.get('chat_id')
    
    if not question or not course or not user_id:
        return jsonify({"error": "Missing query text, course, or user_id"}), 400
        
    start_time = time.time()
    """ 
    local_chat_history = []
    if chat_id:
        chat_id_obj = ObjectId(chat_id)
        chat_doc = course_db.Chats.find_one({"_id": chat_id_obj})
        if chat_doc:
            messages = chat_doc.get("messages", [])
            for i in range(0, len(messages) - 1, 2):
                if i + 1 < len(messages):
                    local_chat_history.append((messages[i]["message_content"], messages[i + 1]["message_content"]))
    """
    transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    question_embedding = transformer_model.encode(question, convert_to_tensor=True)
    question_embedding = question_embedding.cpu().numpy().astype("float32")
    faiss.normalize_L2(question_embedding.reshape(1, -1))
    
    # Load FAISS index and metadata
    index = faiss.read_index(V_DB_PATH +"\\" + course+"_faiss_index.idx")
    with open(V_DB_PATH +"\\" + course+"_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    # Perform a search: retrieve top 3 most similar chunks
    k = 3
    distances, indices = index.search(question_embedding.reshape(1, -1), k)
    
    context_chunks = []
    sources = []  # This will store dictionaries with file and page info.
    for dist, idx in zip(distances[0], indices[0]):
        if dist >= 0.3:
            meta = metadata[idx]
            context_chunks.append(meta["chunk"])
            # Append source information: file name and page number.
            sources.append({
                "pdf": meta.get("pdf", "Unknown"),
                "page": meta.get("page", "Unknown")
            })

    if not context_chunks:
        end_time = time.time()
        return jsonify({
            "response": "I'm sorry, I don't have enough information to answer that.",
            "sources": [],
            "execution_time": f"{end_time - start_time:.2f} seconds"
        })
    
    context = "\n\n".join(context_chunks)

    content_prompt_obj = ChatPromptTemplate.from_template(CONTENT_PROMPT)
    prompt = content_prompt_obj.format(
        context=context, 
        question=question 
    )
    
    response_text = llm_model.invoke(prompt).content
        
    end_time = time.time()
    user_message = {
        "sender": "user",
        "message_content": question,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    chatbot_message = {
        "sender": "chatbot",
        "message_content": response_text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sources": sources
    }
    if chat_id:
        course_db.Chats.update_one(
            {"_id": ObjectId(chat_id)},
            {"$push": {"messages": {"$each": [user_message, chatbot_message]}},
             "$set": {"last_message_time": datetime.now(timezone.utc).isoformat()}}
        )
    else:
        chat_entry = {
            "user_id": user_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_message_time": datetime.now(timezone.utc).isoformat(),
            "messages": [user_message, chatbot_message],
            "title": []
        }
        result = course_db.Chats.insert_one(chat_entry)
        chat_id = str(result.inserted_id)
    return jsonify({
        "chat_id": chat_id,
        "model_response": response_text,
        "sources": sources,
        "execution_time": f"{end_time - start_time:.2f} seconds"
    }), 200