"""THIS FUNCTION GETS ALL CHATS FOR A USER"""
from flask import request, jsonify
from bson import ObjectId

# Helper Function to Get Relevant Data
def extract_fields(data):
    return [
        {
            "_id": entry["_id"],
            "title": entry["title"],
            "last_message_time": entry["last_message_time"]
        }
        for entry in data
    ]

def get_user_chats(db):
    try:
        user_name = request.args.get('user_name')  # Get 'user_name' from query parameters

        if not user_name:
            return jsonify({"error": "Username is required"}), 400

        user = db.Users.find_one({"email": f"{user_name}@sabanciuniv.edu"})
        if not user:
            return jsonify({"error": "User not found"}), 404


        # Find all chats where the user is a participant
        chats = list(db.Chats.find({"user_id": str(user["_id"])}))
        
        chats = extract_fields(chats)

        # Convert ObjectId to string and prepare chats for JSON serialization
        for chat in chats:
            chat['_id'] = str(chat['_id'])
        
       
        # Create a sorted list of chat IDs with their last message times
        chats.sort(key=lambda x: x['last_message_time'], reverse=True)

        return jsonify({
            "chats": chats
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500