# coding=utf-8

from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from Detection import detect
# ... other import statements ...

# creating the Flask application
app = Flask(__name__)
CORS(app)



@app.route('/predictions')
def get_predictions():
    predictions = detect()
    response = make_response(str(predictions), 200)
    response.mimetype = "text/plain"
    print(response)
    return response

@app.route('/statistics', methods=['POST'])
def add_exam():

    return "Viewing statistics", 201
