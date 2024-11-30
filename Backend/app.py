from flask import Flask, request, jsonify
import time
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from Model.vector_database import create_vector_database





app = Flask(__name__)

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


@app.route('/query', methods=['POST'])
def query():
    query_text = request.json.get('query_text')
    if not query_text:
        return jsonify({"error": "No query text provided"}), 400

    start_time = time.time()
    
    # Run create_database.py to generate/update vector database
    #First run vector_database.py create_vector_database()

    embedding_function = OllamaEmbeddings(model="llama3.1:latest")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0:
        return jsonify({"error": "Unable to find matching results"}), 404
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context="", question=query_text)#context_text, question=query_text)

    model = OllamaLLM(model="llama3.1:latest")
    response_text = model.invoke(query_text)

    sources = [doc.metadata.get("source", None) for doc, _score in results]

    end_time = time.time()
    return jsonify({
        "response": response_text,
        "sources": sources,
        "execution_time": f"{end_time - start_time:.2f} seconds"
    })


if __name__ == "__main__":
    app.run()