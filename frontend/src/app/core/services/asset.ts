import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';
import { Asset, PricePoint, WatchlistItem } from '../models';

@Injectable({ providedIn: 'root' })
export class AssetService {
  private base = `${environment.apiUrl}/assets`;

  constructor(private http: HttpClient) {}

  getAssets(): Observable<Asset[]> {
    return this.http.get<Asset[]>(`${this.base}/`);
  }

  getWatchlist(): Observable<WatchlistItem[]> {
    return this.http.get<WatchlistItem[]>(`${this.base}/watchlist/`);
  }

  addToWatchlist(assetId: string): Observable<WatchlistItem> {
    return this.http.post<WatchlistItem>(`${this.base}/watchlist/`, { asset_id: assetId });
  }

  removeFromWatchlist(id: string): Observable<void> {
    return this.http.delete<void>(`${this.base}/watchlist/${id}/`);
  }

  getPriceHistory(symbol: string, days = 30): Observable<PricePoint[]> {
    return this.http.get<PricePoint[]>(`${this.base}/${symbol}/prices/`, { params: { days } });
  }
}
