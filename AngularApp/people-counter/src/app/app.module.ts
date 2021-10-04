import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HourlyPresenceComponent } from './hourly-presence/hourly-presence.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { NgxChartsModule } from '@swimlane/ngx-charts';
import { StartScreenComponent } from './start-screen/start-screen.component';
import {MatButtonModule} from '@angular/material/button';
import { MatSliderModule } from '@angular/material/slider';
import { HttpClientModule } from '@angular/common/http';
import {DetectionService} from "./services/reqest.service";
import {TextFieldModule} from '@angular/cdk/text-field';
import {MatFormFieldModule, MatLabel} from '@angular/material/form-field';
import {MatInputModule} from "@angular/material/input";
import { FormsModule, ReactiveFormsModule  } from '@angular/forms';
import {MatMenuModule} from '@angular/material/menu';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';

@NgModule({
  declarations: [
    AppComponent,
    HourlyPresenceComponent,
    StartScreenComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    AppRoutingModule,
    NgxChartsModule,
    BrowserAnimationsModule,
    MatButtonModule,
    MatSliderModule,
    TextFieldModule,
    MatFormFieldModule,
    MatInputModule,
    MatMenuModule,
    FormsModule,
    ReactiveFormsModule,
    MatProgressSpinnerModule
  ],
  providers: [DetectionService],
  bootstrap: [AppComponent]
})
export class AppModule { }
