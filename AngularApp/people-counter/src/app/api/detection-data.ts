export interface DetectedObject {
  objectClass:string,
  objectClassConfidence:number,
  boundingBoxIOU:number
}

export interface DetectionState {
  isOngoing:string
}


export interface DiscreteDetection{
  detections:DetectedObject[],
  detectionDay:string,
  detectionTime:string
}


export interface DetectionData{
  networkType:string,
  objThreshold:number,
  iouThreshold:number,
  startTime:string,
  numberOfSecondsForDetection:number
}

export interface DetectionDataResponse{
  timestamp:string,
  secondsOfDetection:string,
  detections:any[],
  numberOfDetections:string
}

