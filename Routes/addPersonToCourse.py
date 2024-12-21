"""THIS FUNCTION ADDS A PERSON (INSTRUCTOR/ASSISTANT) TO A COURSE"""
from connectToDB import connect_to_database
from flask import request, jsonify
from bson import ObjectId

def add_person_to_course():
    db = connect_to_database()
    try:

        # These will be obtained from raw JSON body
        user_id = request.json.get('user_id')
        course_code = request.json.get('course_code')

        if not user_id or not course_code:
            return jsonify({"error": "User ID and course_code are required"}), 400
        
        # Find the user by user_id
        user = db.Users.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"error": "User not found"}), 404
       
        course = db.Courses.find_one({"courseCode": course_code})
        if not course:
            return jsonify({"error": "Course not found"}), 404
        
        # Add the course code to user's auth_courses array
        db.Users.update_one(
            {"_id": ObjectId(user_id)},
            {"$addToSet": {"auth_courses": course_code}} # No duplicates allowed
        )

        # Add the user_id to course's personnel array
        db.Courses.update_one(
            {"courseCode": course_code},
            {"$addToSet": {"personnel_ids": user_id}} # No duplicates allowed
        )

        return jsonify({
            "status": "success",
            "message": f"{user['name']} has been added to {course_code}"
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500