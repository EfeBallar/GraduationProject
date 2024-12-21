"""THIS FUNCTION GETS ALL CHATS FOR A USER, GIVEN THEIR EMAIL"""
from flask import request, jsonify
from connectToDB import connect_to_database

def get_user_chats():
    db = connect_to_database()
    try:
        email = request.json.get('email')
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # First, find the user by email to get their user_id
        user = db.Users.find_one({"email": email})
        if not user:
            return jsonify({"error": "User not found"}), 404

        user_id = str(user['_id'])

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