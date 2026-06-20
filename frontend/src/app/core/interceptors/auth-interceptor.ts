import {
  HttpInterceptorFn,
  HttpErrorResponse,
  HttpClient,
  HttpRequest,
  HttpHandlerFn,
} from '@angular/common/http';
import { inject } from '@angular/core';
import { Router } from '@angular/router';
import {
  BehaviorSubject,
  Observable,
  catchError,
  filter,
  switchMap,
  take,
  throwError,
} from 'rxjs';
import { environment } from '../../../environments/environment';

const REFRESH_URL = `${environment.apiUrl}/auth/token/refresh/`;

let isRefreshing = false;
const refreshSubject = new BehaviorSubject<string | null>(null);

function attachToken(req: HttpRequest<unknown>, token: string | null): HttpRequest<unknown> {
  return token ? req.clone({ setHeaders: { Authorization: `Bearer ${token}` } }) : req;
}

function forceLogout(router: Router): void {
  localStorage.removeItem('access_token');
  localStorage.removeItem('refresh_token');
  isRefreshing = false;
  router.navigate(['/login']);
}

function handle401(
  req: HttpRequest<unknown>,
  next: HttpHandlerFn,
  http: HttpClient,
  router: Router,
  originalError: HttpErrorResponse,
): Observable<any> {
  if (req.url === REFRESH_URL) {
    forceLogout(router);
    return throwError(() => originalError);
  }

  if (isRefreshing) {
    return refreshSubject.pipe(
      filter((token): token is string => token !== null),
      take(1),
      switchMap((token) => next(attachToken(req, token))),
    );
  }

  const refreshToken = localStorage.getItem('refresh_token');
  if (!refreshToken) {
    forceLogout(router);
    return throwError(() => originalError);
  }

  isRefreshing = true;
  refreshSubject.next(null);

  return http.post<{ access: string }>(REFRESH_URL, { refresh: refreshToken }).pipe(
    switchMap((res) => {
      localStorage.setItem('access_token', res.access);
      isRefreshing = false;
      refreshSubject.next(res.access);
      return next(attachToken(req, res.access));
    }),
    catchError((err) => {
      forceLogout(router);
      return throwError(() => err);
    }),
  );
}

export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const router = inject(Router);
  const http = inject(HttpClient);
  const token = localStorage.getItem('access_token');
  const authReq = attachToken(req, token);

  return next(authReq).pipe(
    catchError((err: HttpErrorResponse) => {
      if (err.status === 401) {
        return handle401(req, next, http, router, err);
      }
      return throwError(() => err);
    }),
  );
};
