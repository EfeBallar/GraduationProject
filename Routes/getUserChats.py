"""THIS FUNCTION GETS ALL CHATS FOR A USER"""
from flask import request, jsonify
from bson import ObjectId
from connectToDB import connect_to_database

def get_user_chats():
    db = connect_to_database()
    try:
        user_id = request.json.get('user_id')
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        user = db.Users.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Find all chats where the user is a participant
        chats = list(db.Chats.find({"user_id": user_id}))
        
        # Create a list of chat IDs with their last message times
        chatsIDs_with_last_message_time = [
            {
            "chat_id": str(chat['_id']),
            "last_message_time": chat.get('last_message_time')
            }
            for chat in chats
        ]

        return jsonify({
            "chats": chatsIDs_with_last_message_time
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500