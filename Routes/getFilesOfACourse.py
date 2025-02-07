"""THIS FUNCTION RETURNS THE FILES OF A COURSE"""
from flask import request, jsonify
import os
from dotenv import load_dotenv

load_dotenv()
DOC_PATH = os.getenv("DOC_PATH")

def get_files_of_a_course(db):
    try:
        course_code = request.json.get('course_code')

        course = db.Courses.find_one({"courseCode": course_code})
        
        if not course:
            return jsonify({"error": "Course not found"}), 404
        
        file_path = os.path.abspath(os.path.join(DOC_PATH,course_code))

        if not os.path.exists(file_path):
            return jsonify({"files" : [], "file_count" : 0, "msg" : f"There is no path created for {course_code} in the database."}), 404

        items = os.listdir(file_path)
    
        files = [item for item in items if os.path.isfile(os.path.join(file_path, item))]
        
        files = [file for file in files]

        files.sort()
        
        if (len(files) == 0):
            return jsonify({"files" : [], "file_count" : 0, "msg" : "There are no files for this course in database."}), 204

        return jsonify({"files" : files, "file_count" : len(files), "msg" : f"There are {len(files)} file(s) for this course in database."}), 200


    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500