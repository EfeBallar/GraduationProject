from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from pymongo.mongo_client import MongoClient
from langchain_ollama.llms import OllamaLLM
from flask import Flask, request, jsonify
from pymongo.server_api import ServerApi
from datetime import datetime, timezone
from langchain_chroma import Chroma
from dotenv import load_dotenv
from bson import ObjectId
import time
import os

app = Flask(__name__)

# Get info from .env file
load_dotenv()

# Mongo Related Variables
USERNAME = os.getenv("DB_USERNAME")
PASSWORD = os.getenv("PASSWORD")
CLUSTER_ADDRESS = os.getenv("CLUSTER_ADDRESS")
APP_NAME = os.getenv("APP_NAME")

# LLM Related Variables
MODEL = os.getenv("MODEL")


# RAG Related Variables
CHROMA_PATH = os.getenv("CHROMA_PATH")
THRESHOLD = 0.3

uri = f"mongodb+srv://{USERNAME}:{PASSWORD}@{CLUSTER_ADDRESS}/?retryWrites=true&w=majority&appName={APP_NAME}"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Reference to the database and collection
course_db = client.CourseAssistantDB
chats_collection = course_db.Chats


embedding_function = OllamaEmbeddings(model=MODEL)
model = OllamaLLM(model=MODEL)


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}

If no context is provided, kindly inform the user that you cannot answer the question since it is not related with school-related work.
"""
#http://localhost:5000/course/query
@app.route('/course/query', methods=['POST'])
def query():
    chat_course = request.json.get('course')
    query_text = request.json.get('query_text')
    user_id = request.json.get('user_id')
    chat_id = request.json.get('chat_id')
    

    if not query_text or not chat_course or not user_id:
        return jsonify({"error": "Missing query text, course, or user_id"}), 400
    

    chroma_path = f"{CHROMA_PATH}/{chat_course}"
    # print(chroma_path)
    start_time = time.time()
    
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(query_text)

    filtered_results = [ (doc, score) for doc, score in results if score >= THRESHOLD ]

    if len(filtered_results) == 0:
        return jsonify({"error": "Unable to find matching results"}), 404
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

  
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    response_text = model.invoke(prompt)

    sources = list(set(doc.metadata.get("source", None) for doc, _score in results))

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
    # print("here3")


    if (chat_id):   # the chat has been previously created, append to chat history        
        existing_chat = chats_collection.find_one({"_id": ObjectId(f"{chat_id}")})
        if (existing_chat): # prevents problems if random chat id is given somehow
            chats_collection.update_one(
                {"user_id": user_id},
                {
                    "$push": {"messages": {"$each": [user_message, chatbot_message]}},
                    "$set": {"last_message_time": datetime.now(timezone.utc).isoformat()}
                }
            )
        else:
            return jsonify({"error": "Bad chat history ID"}), 400 
    else:   # Create a new chat entry
        chat_entry = {
            "user_id": user_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_message_time": datetime.now(timezone.utc).isoformat(),
            "messages": [user_message, chatbot_message]
        }
        chats_collection.insert_one(chat_entry)

    
    return jsonify({
        "response": response_text,
        "sources": sources,
        "execution_time": f"{end_time - start_time:.2f} seconds"
    })

if __name__ == "__main__":
    app.run()