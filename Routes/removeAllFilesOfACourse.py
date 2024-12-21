"""THIS FUNCTION REMOVES ALL FILES OF A COURSE"""
from flask import request, jsonify
from vector_database import create_vector_database
import os

def remove_all_files_from_course(course_db):
    try:
        # These will be obtained from raw JSON body
        course_code = request.json.get('course_code')

        if not course_code:
            return jsonify({"error": "Course code is required"}), 400
        
        course = course_db.Courses.find_one({"courseCode": course_code})
        
        if not course:
            return jsonify({"error": "Course not found"}), 404
       
        course_data_path = os.path.join('data', 'F24-25', course_code)
        if os.path.exists(course_data_path):
            try:
                for file in os.listdir(course_data_path):
                    os.remove(os.path.join(course_data_path, file))

            except OSError as e:
                return jsonify({"error": f"Error removing file: {str(e)}"}), 500

        else:
            return jsonify({"error": "Course path in data doesn't exist."}), 404

        chroma_data_path = os.path.join('chroma', 'F24-25', course_code)
        if os.path.exists(chroma_data_path):
            try:
                for file in os.listdir(chroma_data_path):
                    os.remove(os.path.join(chroma_data_path, file))

            except OSError as e:
                return jsonify({"error": f"Error removing file: {str(e)}"}), 500

        else:
            return jsonify({"error": "Course path in chroma doesn't exist."}), 404

        return jsonify({
            "status": "success",
            "message": f"All files has been removed from {course_code}"
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500