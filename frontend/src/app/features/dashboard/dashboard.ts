import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { NavigationEnd, Router, RouterLink } from '@angular/router';
import { Subscription, filter, skip } from 'rxjs';
import { AuthService } from '../../core/services/auth';
import { PortfolioService } from '../../core/services/portfolio';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './dashboard.html',
  styleUrl: './dashboard.scss',
})
export class Dashboard implements OnInit, OnDestroy {
  profile: any = null;
  recentJobs: any[] = [];
  loading = true;
  private navSub?: Subscription;

  constructor(
    private authService: AuthService,
    private portfolioService: PortfolioService,
    private router: Router,
  ) {}

  ngOnInit(): void {
    this.loadData();

    this.navSub = this.router.events
      .pipe(
        filter((e): e is NavigationEnd => e instanceof NavigationEnd),
        skip(1),
      )
      .subscribe((e) => {
        if (e.urlAfterRedirects.startsWith('/dashboard')) {
          this.loadData();
        }
      });
  }

  ngOnDestroy(): void {
    this.navSub?.unsubscribe();
  }

  private loadData(): void {
    this.loading = true;
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
