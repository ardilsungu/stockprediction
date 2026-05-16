import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { NavigationEnd, Router, RouterLink } from '@angular/router';
import { Subscription, filter } from 'rxjs';
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
  private inFlight = false;

  constructor(
    private authService: AuthService,
    private portfolioService: PortfolioService,
    private router: Router,
  ) {}

  ngOnInit(): void {
    this.loadData();

    this.navSub = this.router.events
      .pipe(filter((e): e is NavigationEnd => e instanceof NavigationEnd))
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
    if (this.inFlight) return;
    this.inFlight = true;
    this.loading = true;

    let pending = 2;
    const done = () => {
      pending--;
      if (pending === 0) {
        this.inFlight = false;
        this.loading = false;
      }
    };

    this.authService.getProfile().subscribe({
      next: (p) => { this.profile = p; done(); },
      error: () => done(),
    });
    this.portfolioService.getJobs().subscribe({
      next: (jobs) => { this.recentJobs = jobs.slice(0, 3); done(); },
      error: () => done(),
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
