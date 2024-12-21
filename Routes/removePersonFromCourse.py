"""THIS FUNCTION REMOVES A PERSON (INSTRUCTOR/ASSISTANT) FROM A COURSE"""
from flask import request, jsonify
from bson import ObjectId

def remove_person_from_course(db):
    try:

        # These will be obtained from raw JSON body
        user_id = request.json.get('user_id')
        course_code = request.json.get('course_code')

        if not user_id or not course_code:
            return jsonify({"error": "User ID and course_code are required"}), 400
        
        user = db.Users.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"error": "User not found"}), 404
       
        course = db.Courses.find_one({"courseCode": course_code})
        if not course:
            return jsonify({"error": "Course not found"}), 404
        
        # Remove the course code from user's auth_courses array
        db.Users.update_one(
            {"_id": ObjectId(user_id)},
            {"$pull": {"auth_courses": course_code}}
        )

        # Remove the user_id from the course's personnel array
        db.Courses.update_one(
            {"courseCode": course_code},
            {"$pull": {"personnel_ids": user_id}}
        )

        return jsonify({
            "status": "success",
            "message": f"{user['name']} has been removed from {course_code}"
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500