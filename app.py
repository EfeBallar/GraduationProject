from flask import Flask, request, jsonify
import time
from datetime import datetime
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

app = Flask(__name__)

uri = "mongodb+srv://admin:mfB7ViDCuIiBSaeQ@courseassistantdb.j70fx.mongodb.net/?retryWrites=true&w=majority&appName=CourseAssistantDB"
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


MODEL="llama3.1:8b"
embedding_function = OllamaEmbeddings(model=MODEL)
model = OllamaLLM(model=MODEL)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
#http://localhost:5000/course/query
@app.route('/course/query', methods=['POST'])
def query():
    course = request.json.get('course')
    query_text = request.json.get('query_text')
    user_id = request.json.get('user_id')

    if not query_text or not course or not user_id:
        return jsonify({"error": "Missing query text, course, or user_id"}), 400
    
    chroma_path = f"{CHROMA_PATH}/{course}"
    print(chroma_path)
    start_time = time.time()
    
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(query_text)
    if len(results) == 0:
        return jsonify({"error": "Unable to find matching results"}), 404
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("here1")
    response_text = model.invoke(prompt)
    print("here2")
    sources = [doc.metadata.get("source", None) for doc, _score in results]

    end_time = time.time()

    # Construct messages
    user_message = {
        "sender": "user",
        "message_content": query_text,
        "timestamp": datetime.utcnow().isoformat()
    }
    chatbot_message = {
        "sender": "chatbot",
        "message_content": response_text,
        "timestamp": datetime.utcnow().isoformat()
    }
    print("here3")

    # Check if an entry already exists for the user
    existing_chat = chats_collection.find_one({"user_id": user_id})
    print("here4")

    if existing_chat:
        # Append to the existing messages array and update the last_message_time
        chats_collection.update_one(
            {"user_id": user_id},
            {
                "$push": {"messages": {"$each": [user_message, chatbot_message]}},
                "$set": {"last_message_time": datetime.utcnow().isoformat()}
            }
        )
    else:
        # Create a new chat entry
        chat_entry = {
            "user_id": user_id,
            "started_at": datetime.utcnow().isoformat(),
            "last_message_time": datetime.utcnow().isoformat(),
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
    