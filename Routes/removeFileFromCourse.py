"""THIS FUNCTION REMOVES A FILE FROM A COURSE"""
from vector_database import create_vector_database
from flask import request, jsonify
import os

def remove_file_from_course(course_db):
    try:
        # These will be obtained from raw JSON body
        file_name = request.json.get('file_name')
        course_code = request.json.get('course_code')

        if not file_name or not course_code:
            return jsonify({"error": "Course code and file name are required"}), 400
        
        course = course_db.Courses.find_one({"courseCode": course_code})
        
        if not course:
            return jsonify({"error": "Course not found"}), 404
       
        file_path = os.path.join('data', 'F24-25', course_code, file_name)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError as e:
                return jsonify({"error": f"Error removing file: {str(e)}"}), 500

        else:
            return jsonify({"error": "File path doesn't exist."}), 404

        create_vector_database("F24-25", course_code) # Recreate the vector database after removing

        return jsonify({
            "status": "success",
            "message": f"{file_name} has been removed from {course_code}"
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500