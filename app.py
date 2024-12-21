from flask import Flask
from flask_cors import CORS
from connectToDB import connect_to_database
from Routes.query import query
from Routes.getUserChats import get_user_chats
from Routes.getCourses import get_courses
from Routes.addPersonToCourse import add_person_to_course
from Routes.addCourse import add_course
from Routes.removePersonFromCourse import remove_person_from_course
from Routes.getCoursesOfALecturer import get_courses_of_a_lecturer
from Routes.getUserID import get_user_id
from Routes.getPersonnelFromCourse import get_personnel_from_course
from Routes.removeFileFromCourse import remove_file_from_course
from Routes.addFileToCourse import add_file_to_course
from Routes.removeAllFilesOfACourse import remove_all_files_from_course
from Routes.getFilesOfACourse import get_files_of_a_course

app = Flask(__name__) #http://localhost:5000

db = connect_to_database()

# CORS Policy
CORS(app, origins=["http://localhost:3000"])

##################### GET Routes #####################
@app.route('/getUserID', methods=['GET'])
def get_user_id_route():
    return get_user_id(db)

@app.route('/getCourses', methods=['GET'])
def get_courses_route():
    return get_courses(db)

@app.route('/getPersonnelFromCourse', methods=['GET'])
def get_personnel_from_course_route():
    return get_personnel_from_course(db)

@app.route('/getUserChats', methods=['GET'])  
def get_user_chats_route():
    return get_user_chats(db)

@app.route('/getLecturerCourses', methods=['GET'])  
def get_lecturer_courses_route():
    return get_courses_of_a_lecturer(db)

@app.route('/getCourseFiles', methods=['GET'])  
def get_course_files_route():
    return get_files_of_a_course()

##################### POST Routes #####################
@app.route('/<term>/<course>/<chat_id>', methods=['POST'])
@app.route('/<term>/<course>', methods=['POST'])  # Route without chat_id
def query_route(term, course, chat_id=None):
    return query(db, term, course, chat_id)

@app.route('/addFileToCourse', methods=['POST'])
def add_file_to_course_route():
    return add_file_to_course(db) 

@app.route('/addCourse', methods=['POST'])
def add_course_route():
    return add_course(db)

##################### PUT Routes #####################
@app.route('/addPersonToCourse', methods=['PUT'])
def add_person_to_course_route():
    return add_person_to_course(db)

##################### DELETE Routes #####################
@app.route('/removeFileFromCourse', methods=['DELETE'])
def remove_file_from_course_route():
    return remove_file_from_course(db)

@app.route('/removePersonFromCourse', methods=['DELETE'])
def remove_person_from_course_route():
    return remove_person_from_course(db)

@app.route('/removeAllFilesFromCourse', methods=['DELETE'])
def remove_all_files_from_course_route():
    return remove_all_files_from_course(db) 


if __name__ == "__main__":
    app.run()