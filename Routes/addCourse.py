"""THIS FUNCTION ADDS A COURSE"""
from flask import request, jsonify
from bson import ObjectId

def add_course(db):
    try:

        # These will be obtained from raw JSON body
        instructor_id = request.json.get('instructor_id')
        course_code = request.json.get('course_code')
        course_name = request.json.get('course_name')

        if not instructor_id or not course_code or not course_name:
            return jsonify({"error": "Instructor ID and course code/name are required"}), 400
        
        # Find the user by user_id
        instructor = db.Users.find_one({"_id": ObjectId(instructor_id)})
        if not instructor:
            return jsonify({"error": "User not found"}), 404
        
        # Add the course code to user's auth_courses array
        db.Users.update_one(
            {"_id": ObjectId(instructor_id)},
            {"$addToSet": {"auth_courses": course_code}} # No duplicates allowed
        )

        # Add the course to the Courses collection
        db.Courses.insert_one(
            {                
                "courseName": course_name,
                "courseCode": course_code,
                "personnel_ids": [instructor_id]
            }
        )

        return jsonify({
            "status": "success",
            "message": f"{course_code} has been created and added to {instructor['name']}'s courses"
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500