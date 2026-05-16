import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { AuthService } from '../../core/services/auth';
import { PortfolioService } from '../../core/services/portfolio';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './dashboard.html',
  styleUrl: './dashboard.scss',
})
export class Dashboard implements OnInit {
  profile: any = null;
  recentJobs: any[] = [];
  loading = true;

  constructor(
    private authService: AuthService,
    private portfolioService: PortfolioService,
  ) {}

  ngOnInit(): void {
    this.authService.getProfile().subscribe({
      next: (p) => (this.profile = p),
    });
    this.portfolioService.getJobs().subscribe({
      next: (jobs) => {
        this.recentJobs = jobs.slice(0, 3);
        this.loading = false;
      },
      error: () => (this.loading = false),
    });
  }

  statusClass(status: string): string {
    return {
      pending:   'badge-pending',
      running:   'badge-running',
      completed: 'badge-completed',
      failed:    'badge-failed',
    }[status] ?? 'badge-pending';
  }

  statusLabel(status: string): string {
    return { pending: 'Bekliyor', running: 'Çalışıyor', completed: 'Tamamlandı', failed: 'Hata' }[status] ?? status;
  }
}
