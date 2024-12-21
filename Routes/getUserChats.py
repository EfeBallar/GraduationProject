"""THIS FUNCTION GETS ALL CHATS FOR A USER"""
from flask import request, jsonify
from bson import ObjectId

def get_user_chats(db):
    try:
        user_id = request.json.get('user_id')
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        user = db.Users.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Find all chats where the user is a participant
        chats = list(db.Chats.find({"user_id": user_id}))
        
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