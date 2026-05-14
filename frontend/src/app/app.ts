import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet, RouterLink, RouterLinkActive, Router } from '@angular/router';
import { AuthService } from './core/services/auth';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App implements OnInit {
  profile: any = null;

  constructor(public authService: AuthService, private router: Router) {}

  ngOnInit(): void {
    if (this.authService.isLoggedIn()) {
      this.authService.getProfile().subscribe({
        next: (p) => (this.profile = p),
      });
    }
  }

  logout(): void {
    this.authService.logout().subscribe({
      next: () => {
        this.profile = null;
        this.router.navigate(['/login']);
      },
      error: () => {
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        this.profile = null;
        this.router.navigate(['/login']);
      },
    });
  }
}
