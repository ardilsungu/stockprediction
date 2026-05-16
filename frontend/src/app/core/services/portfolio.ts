import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';
import { PortfolioJob, PortfolioJobDetail, PortfolioJobParams } from '../models';

interface DeleteJobResponse {
  message: string;
}

@Injectable({ providedIn: 'root' })
export class PortfolioService {
  private base = `${environment.apiUrl}/portfolio`;

  constructor(private http: HttpClient) {}

  getJobs(): Observable<PortfolioJob[]> {
    return this.http.get<PortfolioJob[]>(`${this.base}/jobs/`);
  }

  createJob(params: PortfolioJobParams): Observable<PortfolioJob> {
    return this.http.post<PortfolioJob>(`${this.base}/jobs/`, { params });
  }

  getJobDetail(id: string): Observable<PortfolioJobDetail> {
    return this.http.get<PortfolioJobDetail>(`${this.base}/jobs/${id}/`);
  }

  deleteJob(id: string): Observable<DeleteJobResponse> {
    return this.http.delete<DeleteJobResponse>(`${this.base}/jobs/${id}/delete/`);
  }
}
