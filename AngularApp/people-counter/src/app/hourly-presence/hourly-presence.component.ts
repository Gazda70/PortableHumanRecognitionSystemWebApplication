import { Component, OnInit } from '@angular/core';
import {NgxChartsModule} from "@swimlane/ngx-charts";
import {DetectionService} from "../services/reqest.service";
import {DetectionData, DetectionDataResponse} from "../api/detection-data";
import { NullVisitor } from '@angular/compiler/src/render3/r3_ast';
import { Subject } from 'rxjs';

@Component({
  selector: 'app-hourly-presence',
  templateUrl: './hourly-presence.component.html',
  styleUrls: ['./hourly-presence.component.css']
})
export class HourlyPresenceComponent implements OnInit {

  constructor(private detectionService:DetectionService) { 
    this.getDetectionData();
  }

  detectionData:any = null;


  public detectionsDurationChartData:any = [];

  xAxisLabelDuration = 'Session number';
  yAxisLabelDuration = 'Session duration';

  public detectionPersonPerFramerChartData:any = [];

  xAxisLabelPerson = 'Number of frame';
  yAxisLabelPerson = 'Number of people on frame';


  testString:string = 'Not obtained test string from server !';


  getDetectionData() {
    this.detectionService.getDetectionStatistics()
      .subscribe(
        (data) => {console.log(data); 
          this.detectionData = JSON.parse(data);
          this.initializeDetectionsDurationChart();
          this.initializeDetectionsPersonNumberChart();
        }
      );
  }

  initializeDetectionsDurationChart(){
    let detDat = [];
    for(var i = 0; i < this.detectionData.length; i++) {
      var obj = this.detectionData[i];
      detDat.push({
        //"name": this.detectionData[i]["timestamp"],
        "name": i,
        "value": this.detectionData[i]["secondsOfDetection"]
      })
  }
    this.detectionsDurationChartData = detDat
  }

  initializeDetectionsPersonNumberChart(){
    var lastDetection = this.detectionData[this.detectionData.length - 1];
    let detFrames = lastDetection["detections"];
    let detFramesChart = [];
    for(var i = 0; i < detFrames.length; i++) {
      console.log("detFrames[i]");
      console.log(detFrames[i]);
        detFramesChart.push({
          "name":i,
          "value": detFrames[i].split('=')[3].split('{').length - 1
        })
    }
    this.detectionPersonPerFramerChartData = detFramesChart;
  }

  obtainDetectionData(){
    return this.detectionsDurationChartData;
  }



  ngOnInit(): void {
    //this.getDetectionData();
  }

  title = 'Angular Charts';

  view: [number, number] = [600, 400];

  // options for the chart
  showXAxis = true;
  showYAxis = true;
  gradient = false;
  showLegend = false;
  showXAxisLabel = true;
  xAxisLabel = 'Session number';
  showYAxisLabel = true;
  yAxisLabel = 'Session duration';
  timeline = true;

  colorScheme = {
    domain: ['#9370DB', '#87CEFA', '#FA8072', '#FF7F50', '#90EE90', '#9370DB']
  };

  //pie
  showLabels = true;

  // data goes here
  /*public sessions = [
    {
      "name": "Monday",
      "value": 2243772
    },
    {
      "name": "Tuesday",
      "value": 1126000
    },
    {
      "name": "Wednesday",
      "value": 296215
    },
    {
      "name": "Thursday",
      "value": 257363
    },
    {
      "name": "Friday",
      "value": 196750
    },
    {
      "name": "Saturday",
      "value": 204617
    },
    {
      "name": "Sunday",
      "value": 45678
    }
  ];*/

  public multi = [
    {
      "name": "China",
      "series": [
        {
          "name": "2018",
          "value": 2243772
        },
        {
          "name": "2017",
          "value": 1227770
        }
      ]
    },

    {
      "name": "USA",
      "series": [
        {
          "name": "2018",
          "value": 1126000
        },
        {
          "name": "2017",
          "value": 764666
        }
      ]
    },

    {
      "name": "Norway",
      "series": [
        {
          "name": "2018",
          "value": 296215
        },
        {
          "name": "2017",
          "value": 209122
        }
      ]
    },

    {
      "name": "Japan",
      "series": [
        {
          "name": "2018",
          "value": 257363
        },
        {
          "name": "2017",
          "value": 205350
        }
      ]
    },

    {
      "name": "Germany",
      "series": [
        {
          "name": "2018",
          "value": 196750
        },
        {
          "name": "2017",
          "value": 129246
        }
      ]
    },

    {
      "name": "France",
      "series": [
        {
          "name": "2018",
          "value": 204617
        },
        {
          "name": "2017",
          "value": 149797
        }
      ]
    }
  ];


  onSelect($event: any) {

  }
}
