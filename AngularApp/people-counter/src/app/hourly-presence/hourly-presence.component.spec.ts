import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HourlyPresenceComponent } from './hourly-presence.component';

describe('HourlyPresenceComponent', () => {
  let component: HourlyPresenceComponent;
  let fixture: ComponentFixture<HourlyPresenceComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ HourlyPresenceComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(HourlyPresenceComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
