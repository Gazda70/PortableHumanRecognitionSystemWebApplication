import cv2
from threading import Thread
from video_stream import VideoStream
from datetime import datetime
import datetime
import time
import tensorflow as tf
import numpy as np
from yolo_functions import OutputRescaler, ImageReader, find_high_class_probability_bbox, nonmax_suppression, ANCHORS, TRUE_BOX_BUFFER


class DetectionManager:
    def __init__(self):
        self.PATH_TO_DETECTION_FILES = ""
        self.SSD_INFERENCE_GRAPH = '/home/pi/Desktop/PeopleCounting/RPIObjectDetection/Code/Detection/SSD/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
        self.SSD_PBTXT = '/home/pi/Desktop/PeopleCounting/RPIObjectDetection/Code/Detection/SSD/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'

    def startDetection(self, neuralNetworkType, detectionSeconds,  obj_threshold=0.3, iou_threshold = 0.1):
        self.neuralNetworkType = neuralNetworkType
        self.obj_threshold = obj_threshold
        self.iou_threshold = iou_threshold
        self.detectionSeconds = detectionSeconds

        detections = self.detect()

    def load_model_SSD(self):
        self.model_SSD = cv2.dnn.readNetFromTensorflow(
            self.SSD_INFERENCE_GRAPH,
            self.SSD_PBTXT)

    def detect_SSD(self, image, img_w, img_h):
        frame = image.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (img_w, img_h))
        self.model_SSD.setInput(cv2.dnn.blobFromImage(frame_resized, size=(img_w, img_h), swapRB=True))
        output = self.model_SSD.forward()
        return output

    def load_model_GazdaWitekLipka(self):
        MODEL_2_TFLITE = "/home/pi/Desktop/PeopleCounting/RPIObjectDetection/TFLite/2021-09-02_2_/model.tflite"
        self.model_GazdaWitekLipka = tf.lite.Interpreter(MODEL_2_TFLITE)
        self.model_GazdaWitekLipka.allocate_tensors()

    def detect_GazdaWitekLipka(self, image, img_w, img_h):
        frame = image.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (img_w, img_h))
        input_details = self.model_GazdaWitekLipka.get_input_details()
        output_details = self.model_GazdaWitekLipka.get_output_details()
        #imageReader = ImageReader(img_h, IMAGE_W=img_w, norm=lambda image: image / 255.)
        #out = imageReader.fit(frame_resized)
        X_test = np.expand_dims(frame_resized, 0).astype('float32')
        dummy_array = np.ones((1, 1, 1, 1, TRUE_BOX_BUFFER, 4)).astype('float32')
        self.model_GazdaWitekLipka.set_tensor(input_details[0]['index'], dummy_array)
        self.model_GazdaWitekLipka.set_tensor(input_details[1]['index'], X_test)
        self.model_GazdaWitekLipka.invoke()
        y_pred = self.model_GazdaWitekLipka.get_tensor(output_details[0]['index'])

        return y_pred

    def detect(self):
        IMAGE_W = 416
        IMAGE_H = 416

        # Initialize video stream
        videostream = VideoStream(resolution=(IMAGE_W, IMAGE_H), framerate=30).start()
        time.sleep(1)

        # Create window
        cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)

        frame = videostream.read()

        start_time = time.time()
        while True:
            netout = []
            if self.neuralNetworkType == "CUSTOM":
                netout = self.detect_GazdaWitekLipka(frame, IMAGE_W, IMAGE_H)
            elif self.neuralNetworkType == "SSD":
                netout = self.detect_SSD(frame, 300, 300)
            else:
                pass

            outputRescaler = OutputRescaler(ANCHORS=ANCHORS)
            netout_scale = outputRescaler.fit(netout)

            boxes = find_high_class_probability_bbox(netout_scale, self.obj_threshold)

            iou_threshold = 0.1
            final_boxes = nonmax_suppression(boxes, iou_threshold=iou_threshold, obj_threshold=self.obj_threshold)

            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time > self.detectionSeconds:
                break

        cv2.destroyAllWindows()
        videostream.stop()

        return final_boxes



    def determineSecondsForDetection(self, detectionTimeString):
        timeValues = []
        timeValues = detectionTimeString.split(':')
        for timeVal in timeValues:
            print("Time value: " + timeVal)
        timeNow = datetime.datetime.now().time()
        hours = int(timeValues[0]) - timeNow.hour
        minutes = int(timeValues[1]) - timeNow.minute
        print("Hours: " + str(hours))
        print("Minutes: " + str(minutes))
        if minutes < 0:
            minutes = 60 - minutes
            hours -= 1

        if hours < 0:
            print("End time must be grater that start time !")
        start_time = time.time()


    def writeDetectionPeriodSummary(self, timestamp):
        f = open(self.PATH_TO_DETECTION_FILES + timestamp, "a")
        f.write("Framerates:\n")
        for i in frame_rate_table:
            f.write(str(i) + "\n")
        f.write("NUMBER_OF_DETECTIONS: " + str(NUMBER_OF_DETECTIONS))
        f.close()