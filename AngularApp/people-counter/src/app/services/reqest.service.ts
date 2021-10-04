import { Injectable } from '@angular/core';
import {HttpClient, HttpHeaders} from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';
import {DetectionData, DetectionState, DiscreteDetection, DetectionDataResponse} from "../api/detection-data";

@Injectable()
export class DetectionService {
  constructor(private http: HttpClient) { }

  detectionDataURL = 'http://192.168.0.241:5000/predictions'

  detectionSetupURL = 'http://192.168.0.241:5000/setup'

  detectionStateURL = 'http://192.168.0.241:5000/check_detection_state'


  httpOptions = {
    headers: new HttpHeaders({
      'Content-Type':  'application/json',
      "Access-Control-Allow-Origin": "*",

    } ),responseType: 'text' as 'json'
  };

  setupNewDetection(startTime:string, numberOfSecondsForDetection:number,
                    networkType:string, objThreshold:number,
                    iouThreshold:number): Observable<any>{
      /*const body = JSON.stringify({'id':1, 'time': {'start': startTime, 'end': endTime}});
      console.log("body: ");
      console.log(body);*/
      const body: DetectionData = {
        networkType:"CUSTOM",
        objThreshold:objThreshold,
        iouThreshold:iouThreshold,
        startTime:startTime,
        numberOfSecondsForDetection:numberOfSecondsForDetection
      }

      return this.http.post(this.detectionSetupURL, body);
  }

  getDetectionStatistics():Observable<any>{
    return this.http.get<any>(this.detectionDataURL, this.httpOptions)
  }
  getDetectionState():Observable<string>{
    return this.http.get<string>(this.detectionStateURL, this.httpOptions)
  }

  getTestString() {
    return this.http.get(this.detectionDataURL, this.httpOptions);
  }

}

