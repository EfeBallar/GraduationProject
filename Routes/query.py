from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from connectToDB import connect_to_database
from datetime import datetime, timezone
from langchain_chroma import Chroma
from flask import request, jsonify
from dotenv import load_dotenv
from bson import ObjectId
import time
import os

load_dotenv()

MODEL = os.getenv("MODEL")
CHROMA_PATH = os.getenv("CHROMA_PATH")
THRESHOLD = 0.3

embedding_function = OllamaEmbeddings(model=MODEL)
model = OllamaLLM(model=MODEL)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}

"""

db = connect_to_database()

def query(term, course, chat_id=None):
    
    # These will be obtained from raw JSON body
    query_text = request.json.get('query_text')
    user_id = request.json.get('user_id')

    if not query_text or not course or not user_id:
        return jsonify({"error": "Missing query text, course, or user_id"}), 400
    
    chroma_path = f"{CHROMA_PATH}/{term}/{course}/"

    start_time = time.time()

    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(query_text)

    filtered_results = [(doc, score) for doc, score in results if score >= THRESHOLD]
    
    if len(filtered_results) == 0:
        end_time = time.time()
        return jsonify({
            "response": "Unable to find matching results from course material.",
            "sources": [],
            "execution_time": f"{end_time - start_time:.2f} seconds"
        })

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    response_text = model.invoke(prompt)

    sources = list(set((doc.metadata.get("source", None), doc.metadata.get("page_number", None)) for doc, _score in filtered_results))

    end_time = time.time()

    # Construct messages
    user_message = {
        "sender": "user",
        "message_content": query_text,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    chatbot_message = {
        "sender": "chatbot",
        "message_content": response_text,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    if chat_id:  # the chat has been previously created, append to chat history
        db.Chats.update_one(
            {"_id": ObjectId(f"{chat_id}")},
            {
                "$push": {"messages": {"$each": [user_message, chatbot_message]}},
                "$set": {"last_message_time": datetime.now(timezone.utc).isoformat()}
            }
        )
    else:  # Create a new chat entry
        chat_entry = {
            "user_id": user_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_message_time": datetime.now(timezone.utc).isoformat(),
            "messages": [user_message, chatbot_message]
        }
        db.Chats.insert_one(chat_entry)

    return jsonify({
        "response": response_text,
        "sources": sources,
        "execution_time": f"{end_time - start_time:.2f} seconds"
    })
