import {Component, OnInit} from '@angular/core';
import {DetectionService} from "./services/reqest.service";
import {DetectionData} from "./api/detection-data";

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit{
  title = 'people-counter';

  constructor(private detectionService:DetectionService) { }

  testString:string = 'Not obtained test string from server !';

  getTestString(){
    this.detectionService.getTestString().subscribe( (test:any) => {
      this.testString = test;
      console.log("test.text: " + test)
    });
  }

  ngOnInit(): void {
    //this.getTestString();
  }

}
