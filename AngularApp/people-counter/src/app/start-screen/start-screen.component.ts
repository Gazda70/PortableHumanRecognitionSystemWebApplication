import {Component, ElementRef, OnChanges, OnInit, SimpleChanges, ViewChild} from '@angular/core';
import {DetectionService} from "../services/reqest.service";
import {FormBuilder, FormGroup} from "@angular/forms";
import {DetectionData, DetectionState} from "../api/detection-data";

@Component({
  selector: 'app-start-screen',
  templateUrl: './start-screen.component.html',
  styleUrls: ['./start-screen.component.css']
})
export class StartScreenComponent implements OnInit, OnChanges {

  neuralNetworksAvailable:string[] = []
  neuralNetworkChosen = '';
  isDetecting:boolean = false;
  endTime:string = '';
  numberOfSecondsForDetection:number = 0;

 /* time: number = 0;
  display: any;
  interval: any;

  startTimer() {
    console.log("=====>");
    this.interval = setInterval(() => {
      if (this.time === 0) {
        this.time++;
      } else {
        this.time++;
      }
      this.display=this.transform( this.time)
    }, 1000);
  }
  transform(value: number): string {
    const minutes: number = Math.floor(value / 60);
    return minutes + ':' + (value - minutes * 60);
  }
  pauseTimer() {
    clearInterval(this.interval);
  }*/


  constructor(private detectionService:DetectionService) {
    this.neuralNetworksAvailable.push("GazdaWitekLipka Detector");
    this.neuralNetworksAvailable.push("SSD Mobilenetv2 Detector");
    this.neuralNetworksAvailable.push("YOLO Tiny v2 Detector");
  }
  ngOnChanges(changes: SimpleChanges): void {
    /*this.configService.getDetectionState().subscribe(
      {
        next: (value => {
          if(value == "true"){
            this.isDetecting = true;
          }
          console.log("Is detection: " + value);
        })
      }
    )*/
  }

  ngOnInit(): void {
    //console.log("Start");
    //this.validateDetectionTime();
  }

  detect(){
    const setupCheck = this.validateDetectionSetup();
    console.log("setupCheck: " + setupCheck);
    if(setupCheck == true){
      this.isDetecting = true;
      console.log("Detecting");
      this.detectionService.setupNewDetection('', this.numberOfSecondsForDetection,
      this.convertNetworkName(this.neuralNetworkChosen), 0.3, 0.1).subscribe(
        {
          next: (value => {console.log("Response: " + value);})
        }
      )
    }
  }

  convertNetworkName(nName:string):string{
    if(nName == "GazdaWitekLipka Detector"){
      return "CUSTOM";
    }else if(nName == "SSD Mobilenetv2 Detector"){
      return "SSD";
    }else{
      return "YOLO";
    }
  }

  setChosenNeuralNetwork(neuralNetworkName:string){
    this.neuralNetworkChosen = neuralNetworkName;
  }

  validateDetectionTime():boolean{
    var re = /^(?:([01]?\d|2[0-3]):([0-5]?\d))?$/;
    if(re.test(this.endTime)){
      const hhMM = this.endTime.split(':');
      var currentdate = new Date();
      var resultHours = parseInt(hhMM[0], 10) - currentdate.getHours();
      var resultMinutes = parseInt(hhMM[1], 10) - currentdate.getMinutes();
      if (resultHours < 0){
          window.alert("You can't setup detection for time in past. Increase hour !");
          return false;
      }
      if(resultMinutes < 0) {
        if(resultHours == 0){
          window.alert("You can't setup detection for time in past. Increase minutes !");
          return false;
        }
        resultMinutes = 60 - resultMinutes;
      }
      console.log("Result hours: " + resultHours);
      console.log("Result minutes: " + resultMinutes);
      this.calculateSecondsForDetection(resultHours, resultMinutes);
      console.log("Seconds for detection: " + this.numberOfSecondsForDetection);
      return true;
    }else{
      window.alert("You need to put detection end time in the format of HH:MM !");
      return false;
    }
  }

  validateDetectionSetup():boolean{
    if(this.endTime != '') {
      const timeCheck = this.validateDetectionTime();
      if(timeCheck == true){
        console.log("timeCheck: " + timeCheck);
        return true;
      }
    }else{
      window.alert("You didnt't fill out all necessary forms to start detecting !");
    }
    return false;
  }

  calculateSecondsForDetection(hours:number, minutes:number){
    this.numberOfSecondsForDetection = hours * 3600 + minutes * 60;
  }

}
