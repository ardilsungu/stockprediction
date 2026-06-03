import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { ActivatedRoute, convertToParamMap } from '@angular/router';
import { of, Subject } from 'rxjs';
import { vi } from 'vitest';

import { Dashboard } from './dashboard';
import { AuthService } from '../../core/services/auth';
import { PortfolioService } from '../../core/services/portfolio';

describe('Dashboard', () => {
  let fixture: ComponentFixture<Dashboard>;
  let urlSubject: Subject<any>;
  let getProfileSpy: ReturnType<typeof vi.fn>;
  let getJobsSpy: ReturnType<typeof vi.fn>;

  beforeEach(async () => {
    urlSubject = new Subject();
    getProfileSpy = vi.fn().mockReturnValue(of({ username: 'test' }));
    getJobsSpy = vi.fn().mockReturnValue(of([]));

    await TestBed.configureTestingModule({
      imports: [Dashboard],
      providers: [
        provideRouter([]),
        { provide: AuthService, useValue: { getProfile: getProfileSpy } },
        { provide: PortfolioService, useValue: { getJobs: getJobsSpy } },
        {
          provide: ActivatedRoute,
          useValue: {
            url: urlSubject.asObservable(),
            snapshot: { paramMap: convertToParamMap({}) },
          },
        },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(Dashboard);
  });

  it('should call getProfile and getJobs on ngOnInit', () => {
    fixture.detectChanges(); // ngOnInit çalışır, route.url subscribe olur
    urlSubject.next([]);     // subscribe olduktan sonra emit → callback çalışır

    expect(getProfileSpy).toHaveBeenCalledTimes(1);
    expect(getJobsSpy).toHaveBeenCalledTimes(1);
  });

  it('should reload data when navigating to same URL', () => {
    fixture.detectChanges(); // ngOnInit, subscription kurulur
    urlSubject.next([]);     // ilk emit
    urlSubject.next([]);     // ikinci emit (aynı URL tekrar)

    expect(getProfileSpy).toHaveBeenCalledTimes(2);
    expect(getJobsSpy).toHaveBeenCalledTimes(2);
  });
});
