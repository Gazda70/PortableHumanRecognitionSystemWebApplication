# coding=utf-8

from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from detection_manager import DetectionManager
import json
#from Detection import detect
# ... other import statements ...

# creating the Flask application
app = Flask(__name__)
CORS(app)

detection_manager = DetectionManager()

@app.route('/predictions')
def get_predictions():
    '''
    predictions = detect()
    response = make_response(str(predictions), 200)
    response.mimetype = "text/plain"
    print(response)
    return response
    '''
    print("I am GET")
    return "I am working"

@app.route('/setup', methods=['GET','POST'])
def setup_detection():
    print("I am POST")
    print("Request body: ")
    req = request.get_json()
    '''
    if request.method == 'POST':

    elif request.method == 'GET':
    '''
    detection_manager.startDetection(req["networkType"], req["endTime"],
                                     float(req["objThreshold"]), float(req["iouThreshold"]))
    print(req)
    print("Request part: ")
    print(req['endDay'])
    response_body = {
      "startDay": "Wed",
      "endDay": "Thurs",
      "startTime": "12:34",
      "endTime": "15:20",
      "totalDetections": []
    }
    response_body_json = json.dumps(response_body)
    response = make_response(response_body_json, 200)
    response.mimetype = "application/json"
    print('Response: ')
    print(response)
    return response
