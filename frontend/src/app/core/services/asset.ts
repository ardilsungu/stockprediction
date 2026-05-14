import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';

@Injectable({ providedIn: 'root' })
export class AssetService {
  private base = `${environment.apiUrl}/assets`;

  constructor(private http: HttpClient) {}

  getAssets(): Observable<any[]> {
    return this.http.get<any[]>(`${this.base}/`);
  }

  getWatchlist(): Observable<any[]> {
    return this.http.get<any[]>(`${this.base}/watchlist/`);
  }

  addToWatchlist(assetId: string): Observable<any> {
    return this.http.post<any>(`${this.base}/watchlist/`, { asset_id: assetId });
  }

  removeFromWatchlist(id: string): Observable<any> {
    return this.http.delete<any>(`${this.base}/watchlist/${id}/`);
  }
}
