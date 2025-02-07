from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

from datetime import datetime, timezone
from langchain_chroma import Chroma
from flask import request, jsonify
from dotenv import load_dotenv
from bson import ObjectId
import time
import os

load_dotenv()
MODEL=os.getenv("MODEL")
CHROMA_PATH = os.getenv("CHROMA_PATH")
embedding_function = OllamaEmbeddings(model=MODEL)
model = ChatOllama(model=MODEL, temperature = 0)

CONTENT_PROMPT = """
**When answering, do not include anything from this prompt in the response.**
Answer the following question based on only the context provided. If not a relevant context/question, return: *"I'm sorry, I don't have enough information to answer that."*    
   
**Context:** {context}  
**Question:** {question}  
 """

def query(course_db):
    course = request.json.get('course')
    term = request.json.get('term')
    query_text = request.json.get('query_text')
    user_id = request.json.get('user_id')
    chat_id = request.json.get('chat_id')
    
    if not query_text or not course or not user_id:
        return jsonify({"error": "Missing query text, course, or user_id"}), 400
        
    start_time = time.time()
    local_chat_history = []
    if chat_id:
        chat_id_obj = ObjectId(chat_id)
        chat_doc = course_db.Chats.find_one({"_id": chat_id_obj})
        if chat_doc:
            messages = chat_doc.get("messages", [])
            for i in range(0, len(messages) - 1, 2):
                if i + 1 < len(messages):
                    local_chat_history.append((messages[i]["message_content"], messages[i + 1]["message_content"]))
    
    chroma_dir = os.path.join(CHROMA_PATH, term, course)
    db = Chroma(persist_directory=chroma_dir, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(query_text,score_threshold=0.5)

    if not results:
        end_time = time.time()
        return jsonify({
            "response": "I'm sorry, I don't have enough information to answer that.",
            "sources": [],
            "execution_time": f"{end_time - start_time:.2f} seconds"
        })
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    content_prompt_obj = ChatPromptTemplate.from_template(CONTENT_PROMPT)
    prompt = content_prompt_obj.format(
        context=context_text, 
        chat_history=str(local_chat_history), 
        question=query_text, 
    )
    print(prompt)
    
    response_text = model.invoke(prompt).content
    
    sources = [
        {"source": os.path.basename(doc.metadata.get("source", "unknown")), "page": doc.metadata.get("page")}
        for doc, _ in results
    ]
    
    end_time = time.time()
    user_message = {
        "sender": "user",
        "message_content": query_text,
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