# coding=utf-8

from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
# ... other import statements ...

# creating the Flask application
app = Flask(__name__)
CORS(app)



@app.route('/predicitons')
def get_exams():
    response = make_response("Here you are: predicitons", 200)
    response.mimetype = "text/plain"
    print(response)
    return response

@app.route('/statistics', methods=['POST'])
def add_exam():

    return "Viewing statistics", 201
