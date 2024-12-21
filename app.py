from flask import Flask
from flask_cors import CORS
from Routes.getUserChats import get_user_chats
from Routes.query import query

app = Flask(__name__) #http://localhost:5000

# CORS Policy
CORS(app, origins=["http://localhost:3000"])

@app.route('/<term>/<course>/<chat_id>', methods=['POST'])
@app.route('/<term>/<course>/', methods=['POST'])  # Route without chat_id
def query_route(term, course, chat_id=None):
    return query(term, course, chat_id)

@app.route('/getUserChats/', methods=['GET'])  
def get_user_chats_route():
    return get_user_chats()

if __name__ == "__main__":
    app.run()