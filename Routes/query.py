from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from datetime import datetime, timezone
from langchain_chroma import Chroma
from flask import request, jsonify
from dotenv import load_dotenv
from bson import ObjectId
#from langchain.memory import ConversationBufferMemory #TOD BUNLARA BAK
#from langchain.chains import ConversationalRetrievalChain
import time
import os

load_dotenv()
chat_history = []
MODEL = os.getenv("MODEL")
CHROMA_PATH = os.getenv("CHROMA_PATH")
THRESHOLD = 0.3

embedding_function = OllamaEmbeddings(model=MODEL)
model = OllamaLLM(model=MODEL)

QUERY_CLASSIFIER_PROMPT = """
Determine if this question is about the conversation history itself or about course content.
Question: {question}

Return only one word:
- 'conversation' if the question is about previous messages, what was said before, or the chat history
- 'content' if the question is about course material or any other topic

Answer:"""

CONVERSATION_PROMPT = """
Based on the following chat history, answer the question.
If there's no relevant information in the chat history, say "I don't have enough conversation history to answer that."

Chat History: {chat_history}

Question: {question}

Answer based strictly on the chat history above:"""

CONTENT_PROMPT = """
Answer the question based on the following context and chat history:

Context: {context}

Chat History: {chat_history}

---

Current Question: {question}

Remember to consider both the context and the chat history when forming your answer.
"""
def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    if max_score - min_score == 0:
        return [0.0 for _ in scores]  # Avoid division by zero
    return [(score - min_score) / (max_score - min_score) for score in scores]

def query(course_db):
    course = request.json.get('course')
    term = request.json.get('term') or None
    query_text = request.json.get('query_text')
    user_id = request.json.get('user_id')
    chat_id = request.json.get('chat_id')

    print("Inside query function")
    print(f"term: {term}")
    print(f"course: {course}")
    print(f"query text: {query_text}")
    print(f"user id: {user_id}")
    print(f"chat id: {chat_id}")

    if not query_text or not course or not user_id:
        return jsonify({"error": "Missing query text, course, or user_id"}), 400

    start_time = time.time()

    # Initialize chat_history and handle upsert
    global chat_history
    chat_history = []

    if chat_id:
        chat_id_obj = ObjectId(chat_id)

        # Find existing chat document
        chat_doc = course_db.Chats.find_one({"_id": chat_id_obj})
        """if chat_doc:"""
        # Populate chat history if the document exists
        messages = chat_doc.get("messages", [])
        for i in range(0, len(messages) - 1, 2):
            if i + 1 < len(messages):
                user_msg = messages[i]["message_content"]
                assistant_msg = messages[i + 1]["message_content"]
                chat_history.append((user_msg, assistant_msg))
                #TOD Sourcelar da historyde depolanacak ki gösterebilelim
        """else:
            # Create a new chat document if not found
            initial_chat_doc = {
                "_id": chat_id_obj,
                "user_id": user_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "last_message_time": datetime.now(timezone.utc).isoformat(),
                "messages": [],
                "title": None
            }
            course_db.Chats.insert_one(initial_chat_doc)"""

    # First, classify the query type
    classifier_prompt = ChatPromptTemplate.from_template(QUERY_CLASSIFIER_PROMPT)
    query_type = model.invoke(classifier_prompt.format(question=query_text)).strip().lower()
    print(query_type)
    sources = []
    print(chat_history)
    if query_type == 'conversation' and chat_history:
        # Handle conversation-related query
        conversation_prompt = ChatPromptTemplate.from_template(CONVERSATION_PROMPT)
        prompt = conversation_prompt.format(
            chat_history=str(chat_history),
            question=query_text
        )
        response_text = model.invoke(prompt)

        
    else:
        # Handle content-related query
        chroma_path = f"{CHROMA_PATH}/{term}/{course}/"
        db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
        results = db.similarity_search_with_relevance_scores(query_text, k = 30)
        # Debug: Check score range
        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)
        print(f"Relevance Scores - Min: {min_score}, Max: {max_score}")
        
        # Option 1: Normalize Scores
        normalized_scores = normalize_scores(scores)
        # Option 2: Use Top-K without threshold
        # normalized_scores = scores  # If you prefer to use raw scores

        # Option 1: Filtering with normalized scores
        filtered_results = [
            (doc, norm_score) 
            for (doc, _), norm_score in zip(results, normalized_scores) 
            if norm_score >= THRESHOLD
        ]

        # Option 2: Using Top-K without threshold
        # filtered_results = results[:10]  # Example: top 10 results
        #filtered_results = [(doc, score) for doc, score in results if score >= THRESHOLD]

        # if score >= THRESHOLD
        if len(filtered_results) == 0:
            end_time = time.time()
            return jsonify({
                "response": "Unable to find matching results from course material.",
                "sources": [],
                "execution_time": f"{end_time - start_time:.2f} seconds"
            })

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered_results])
        content_prompt = ChatPromptTemplate.from_template(CONTENT_PROMPT)
        prompt = content_prompt.format(
            context=context_text,
            chat_history=str(chat_history),
            question=query_text
        )
        response_text = model.invoke(prompt)

        sources = [
            {
                "source": os.path.basename(doc.metadata.get("source", "unknown")),
                "page": doc.metadata.get("page", None)
            }
            for doc, _score in filtered_results
        ]

    end_time = time.time()

    # Prepare user and chatbot messages
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

    # Update or create chat document with new messages
    if chat_id:
        course_db.Chats.update_one(
            {"_id": chat_id_obj},
            {
                "$push": {"messages": {"$each": [user_message, chatbot_message]}},
                "$set": {"last_message_time": datetime.now(timezone.utc).isoformat()}
            }
        )
    else:
        title_prompt = ChatPromptTemplate.from_template(
            "Create a concise title (up to 6 words) for this conversation based on the user query:\n\nUser Query: {question}"
        )
        title_input = title_prompt.format(question=query_text)
        title_response = model.invoke(title_input)

        title = title_response.strip().replace('"', '').replace("'", '')

        chat_entry = {
            "user_id": user_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_message_time": datetime.now(timezone.utc).isoformat(),
            "messages": [user_message, chatbot_message],
            "title": title
        }

        result = course_db.Chats.insert_one(chat_entry)
        chat_id = str(result.inserted_id)

    return jsonify({
        "chat_id": chat_id,
        "model_response": response_text,
        "sources": sources,
        "execution_time": f"{end_time - start_time:.2f} seconds"
    }), 200