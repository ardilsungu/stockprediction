import { Routes } from '@angular/router';
import { Login } from './features/auth/login/login';
import { Register } from './features/auth/register/register';
import { Dashboard } from './features/dashboard/dashboard';
import { Watchlist } from './features/watchlist/watchlist';
import { Portfolio } from './features/portfolio/portfolio';
import { authGuard } from './core/guards/auth-guard';

export const routes: Routes = [
  { path: '', redirectTo: '/dashboard', pathMatch: 'full' },
  { path: 'login',     component: Login },
  { path: 'register',  component: Register },
  { path: 'dashboard', component: Dashboard,  canActivate: [authGuard] },
  { path: 'watchlist', component: Watchlist,  canActivate: [authGuard] },
  { path: 'portfolio', component: Portfolio,  canActivate: [authGuard] },
];
