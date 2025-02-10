from langchain.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_ollama import ChatOllama
from datetime import datetime, timezone
from flask import request, jsonify
from dotenv import load_dotenv
from bson import ObjectId
import torch
import faiss
import pickle
import os

load_dotenv()
LLM_MODEL=os.getenv("LLM_MODEL")
V_DB_PATH = os.getenv("V_DB_PATH")
THRESHOLD = float(os.getenv("CHUNK_RELEVANCY_THRESHOLD"))

llm_model = ChatOllama(model=LLM_MODEL, temperature = 0)
# title_model = ChatOllama(model="", temperature = 0)


reasoning_rule = "Respond directly without additional reasoning." if "deepseek" in LLM_MODEL else ""

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

def retrieve_from_vector_database(question,course):
    transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    question_embedding = transformer_model.encode(question, convert_to_tensor=True)
    question_embedding = question_embedding.cpu().numpy().astype("float32")
    faiss.normalize_L2(question_embedding.reshape(1, -1))
    
    # Load FAISS index and metadata
    index = faiss.read_index(V_DB_PATH +"\\" + course+"_faiss_index.idx")
    with open(V_DB_PATH +"\\" + course+"_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    # Perform a search: retrieve top k most similar chunks
    k = 5
    distances, indices = index.search(question_embedding.reshape(1, -1), k)

    context_chunks = []
    sources = []  # This will store dictionaries with file and page info.
    for dist, idx in zip(distances[0], indices[0]):
        if dist >= THRESHOLD:
            meta = metadata[idx]
            context_chunks.append(meta["chunk"])


            # Append source information: file name and page number.
            sources.append({
                "pdf": meta.get("pdf", "Unknown"),
                "page": meta.get("page", "Unknown")
            })
    
    return context_chunks, sources

def query(course_db):
    course = request.json.get('course')
    question = request.json.get('question')
    user_id = request.json.get('user_id')
    chat_id = request.json.get('chat_id')

    if not question or not course or not user_id:
        return jsonify({"error": "Missing query text, course, or user_id"}), 400
        
    context_chunks, sources = retrieve_from_vector_database(question,course)


    if not context_chunks: # If no context is found
        if not chat_id:
            return jsonify({
                "response": "I'm sorry, I don't have enough information to answer that.",
                "sources": []
            })
        
        else: # If there is a chat_id, we can use the chat history to generate a response
            chat_id_obj = ObjectId(chat_id)
            chat_doc = course_db.Chats.find_one({"_id": chat_id_obj})
            if chat_doc:
                last_3_memory = ConversationBufferWindowMemory(k=3)
                messages = chat_doc.get("messages", [])
                for i in range(0, len(messages) - 1, 2):
                    if i + 1 < len(messages):
                        last_3_memory.save_context(
                            {"input": messages[i]["message_content"]},
                            {"output": messages[i + 1]["message_content"]}
                        )
                
                conversation = ConversationChain(
                    llm=llm_model,
                    verbose=True,
                    memory=last_3_memory
                )
                rule = " Please answer the question using only the chat history provided. Do not include or search for any additional knowledge or assumptions beyond what is given. If the answer cannot be determined from the chat history between you and the user, explicitly state that the information is not available."
                response_text = conversation.predict(input=question+rule)
                response_text = response_text[response_text.rfind("AI:") + len("AI:") + 1:]

    else:
        # If context is found, use it to generate a response
        context = "\n\n".join(context_chunks)

        content_prompt_obj = ChatPromptTemplate.from_template(CONTENT_PROMPT)
        prompt = content_prompt_obj.format(
            context=context, 
            question=question 
        )
    
        response_text = llm_model.invoke(reasoning_rule + prompt).content


    if "</think>" in response_text:
        response_text = response_text[response_text.index("</think>") + len("</think>") + 1:]
        
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
        title_prompt = ChatPromptTemplate.from_template(
            "Create a concise title (up to 6 words) for this conversation based on the user query:\n\nUser Query: {question}. Make sure you only return the title itself wrapped in [ and ]. For example, if the generated title is 'Title', then the output will be ['Title']"
        )
        title_input = title_prompt.format(question=question)
        title_response = llm_model.invoke(title_input)


        title = title_response.content[title_response.content.rfind("[") + 1:-1].strip().replace('"', '').replace("'", '')
    
        chat_entry = {
            "course": course,
            "user_id": user_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_message_time": datetime.now(timezone.utc).isoformat(),
            "messages": [user_message, chatbot_message],
            "title": title
        }

        result = course_db.Chats.insert_one(chat_entry)
        chat_id = str(result.inserted_id)
    return jsonify({
        "course": course,
        "chat_id": chat_id,
        "model_response": response_text,
        "sources": sources
    }), 200