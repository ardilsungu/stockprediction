import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, tap } from 'rxjs';
import { environment } from '../../../environments/environment';
import { LoginResponse, LogoutResponse, RegisterResponse, User } from '../models';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private apiUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  login(email: string, password: string): Observable<LoginResponse> {
    return this.http.post<LoginResponse>(`${this.apiUrl}/auth/login/`, { email, password }).pipe(
      tap(res => {
        localStorage.setItem('access_token', res.access);
        localStorage.setItem('refresh_token', res.refresh);
      })
    );
  }

  register(email: string, username: string, password: string): Observable<RegisterResponse> {
    return this.http.post<RegisterResponse>(`${this.apiUrl}/auth/register/`, {
      email,
      username,
      password,
      password_confirm: password,
    });
  }

  logout(): Observable<LogoutResponse> {
    const refresh = localStorage.getItem('refresh_token');
    return this.http.post<LogoutResponse>(`${this.apiUrl}/auth/logout/`, { refresh }).pipe(
      tap(() => {
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
      })
    );
  }

  getProfile(): Observable<User> {
    return this.http.get<User>(`${this.apiUrl}/auth/profile/`);
  }

  getToken(): string | null {
    return localStorage.getItem('access_token');
  }

  isLoggedIn(): boolean {
    return !!this.getToken();
  }
}
