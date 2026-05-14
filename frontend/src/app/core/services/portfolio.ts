import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';

@Injectable({ providedIn: 'root' })
export class PortfolioService {
  private base = `${environment.apiUrl}/portfolio`;

  constructor(private http: HttpClient) {}

  getJobs(): Observable<any[]> {
    return this.http.get<any[]>(`${this.base}/jobs/`);
  }

  createJob(params: Record<string, any>): Observable<any> {
    return this.http.post<any>(`${this.base}/jobs/`, { params });
  }

  getJobDetail(id: string): Observable<any> {
    return this.http.get<any>(`${this.base}/jobs/${id}/`);
  }

  deleteJob(id: string): Observable<any> {
    return this.http.delete<any>(`${this.base}/jobs/${id}/delete/`);
  }
}
