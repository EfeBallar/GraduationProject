"""THIS FUNCTION ADDS A FILE TO A COURSE"""
from flask import request, jsonify
from vector_database import create_vector_database
import os

def add_file_to_course(course_db):
    try:
        # get the PDF file here and put it to fileToAdd variable, key = fileName
        # get the course code here and put it to courseCode variable, key = courseCode 
        file_to_add = request.files.get('fileName')
        course_code = request.form.get('courseCode')  
        
        if not course_code or not file_to_add:
            return jsonify({"error": "File and course_code are required"}), 400
        
        course = course_db.Courses.find_one({"courseCode": course_code})
        
        if not course:
            return jsonify({"error": "Course not found"}), 404
        
        try:
            file_path = os.path.join('data', 'F24-25', course_code)
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            file_to_add.save(os.path.join(file_path, file_to_add.filename)) # Save with the full path
            create_vector_database("F24-25", course_code)

        except:
            return jsonify({"error": "File could not be saved"}), 400
            
        return jsonify({
            "status": "success",
            "message": f"{file_to_add.filename} has been added to {course_code}"
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500