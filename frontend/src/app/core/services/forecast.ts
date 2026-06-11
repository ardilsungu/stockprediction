import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';
import { ForecastJob, ForecastJobDetail } from '../models';

@Injectable({ providedIn: 'root' })
export class ForecastService {
  private base = `${environment.apiUrl}/forecast`;

  constructor(private http: HttpClient) {}

  createJob(
    asset: string,
    model_type: ForecastJob['model_type'],
    horizon_days: number,
  ): Observable<ForecastJob> {
    return this.http.post<ForecastJob>(`${this.base}/jobs/`, { asset, model_type, horizon_days });
  }

  getJobDetail(id: string): Observable<ForecastJobDetail> {
    return this.http.get<ForecastJobDetail>(`${this.base}/jobs/${id}/`);
  }
}
