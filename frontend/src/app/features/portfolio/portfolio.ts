import { Component, OnInit, OnDestroy, AfterViewInit, ElementRef, ViewChild, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { PortfolioService } from '../../core/services/portfolio';
import { ParetoSolution, PortfolioJobDetail, StrategyMetrics } from '../../core/models';
import { createChart, ColorType, LineSeries } from 'lightweight-charts';

interface StrategyRow extends StrategyMetrics {
  key: string;
}

interface WeightRow {
  ticker: string;
  weight: number;
}

@Component({
  selector: 'app-portfolio',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './portfolio.html',
  styleUrl: './portfolio.scss',
})
export class Portfolio implements OnInit, OnDestroy, AfterViewInit {
  @ViewChild('chartContainer') chartContainer!: ElementRef;

  form: FormGroup;
  submitting = false;
  currentJob = signal<PortfolioJobDetail | null>(null);
  selectedStrategy = signal<string | null>(null);

  strategies = computed<StrategyRow[]>(() => {
    const dict = this.currentJob()?.result?.strategies;
    if (!dict) return [];
    return Object.entries(dict).map(([key, metrics]) => ({ key, ...metrics }));
  });

  selectedWeights = computed<WeightRow[]>(() => {
    const key = this.selectedStrategy();
    if (!key) return [];
    const weights = this.currentJob()?.result?.weights?.[key];
    if (!weights) return [];
    return Object.entries(weights)
      .map(([ticker, weight]) => ({ ticker, weight }))
      .sort((a, b) => b.weight - a.weight);
  });

  selectedStrategyName = computed<string | null>(() => {
    const key = this.selectedStrategy();
    if (!key) return null;
    return this.currentJob()?.result?.strategies?.[key]?.name ?? key;
  });

  pollInterval: any = null;
  errorMessage = '';
  private chart: any = null;

  constructor(private fb: FormBuilder, private portfolioService: PortfolioService) {
    this.form = this.fb.group({
      n_coins:      [50,  [Validators.required, Validators.min(20),  Validators.max(100)]],
      lookback_days:[365, [Validators.required, Validators.min(250), Validators.max(750)]],
      n_gen:        [100, [Validators.required, Validators.min(20),  Validators.max(500)]],
      pop_size:     [100, [Validators.required, Validators.min(20),  Validators.max(500)]],
    });
  }

  ngOnInit(): void {}
  ngAfterViewInit(): void {}

  ngOnDestroy(): void {
    this.clearPoll();
    this.chart?.remove();
  }

  startJob(): void {
    if (this.form.invalid || this.submitting) return;
    this.submitting = true;
    this.currentJob.set(null);
    this.selectedStrategy.set(null);
    this.errorMessage = '';
    this.chart?.remove();
    this.chart = null;

    this.portfolioService.createJob(this.form.value).subscribe({
      next: (job) => {
        this.currentJob.set(job as PortfolioJobDetail);
        this.submitting = false;
        this.startPolling(job.id);
      },
      error: () => {
        this.errorMessage = 'Job başlatılamadı.';
        this.submitting = false;
      },
    });
  }

  selectStrategy(key: string): void {
    this.selectedStrategy.set(this.selectedStrategy() === key ? null : key);
  }

  formatPct(v: number | null | undefined): string {
    return v != null ? (v * 100).toFixed(2) + '%' : '—';
  }

  statusClass(status: string): string {
    return { pending: 'badge-pending', running: 'badge-running', completed: 'badge-completed', failed: 'badge-failed' }[status] ?? '';
  }

  statusLabel(status: string): string {
    return { pending: 'Bekliyor', running: 'Çalışıyor...', completed: 'Tamamlandı', failed: 'Hata' }[status] ?? status;
  }

  private startPolling(jobId: string): void {
    this.pollJob(jobId);
    this.pollInterval = setInterval(() => this.pollJob(jobId), 3000);
  }

  private pollJob(jobId: string): void {
    this.portfolioService.getJobDetail(jobId).subscribe({
      next: (job) => {
        this.currentJob.set(job);
        if (job.status === 'completed' || job.status === 'failed') {
          this.clearPoll();
          if (job.status === 'completed') {
            const first = Object.keys(job.result?.strategies ?? {})[0] ?? null;
            if (first && !this.selectedStrategy()) this.selectedStrategy.set(first);
            setTimeout(() => this.renderChart(job.result?.pareto_solutions ?? []), 100);
          }
        }
      },
    });
  }

  private clearPoll(): void {
    if (this.pollInterval) { clearInterval(this.pollInterval); this.pollInterval = null; }
  }

  private renderChart(pareto: ParetoSolution[]): void {
    if (!this.chartContainer?.nativeElement || pareto.length === 0) return;
    this.chart?.remove();

    this.chart = createChart(this.chartContainer.nativeElement, {
      width: this.chartContainer.nativeElement.offsetWidth,
      height: 300,
      layout: { background: { type: ColorType.Solid, color: '#0f1117' }, textColor: '#94a3b8' },
      grid: { vertLines: { color: '#1e2235' }, horzLines: { color: '#1e2235' } },
      rightPriceScale: { borderColor: '#2a2d3e' },
      timeScale: { borderColor: '#2a2d3e', visible: false },
    });

    const series = this.chart.addSeries(LineSeries, {
      color: '#6366f1', lineWidth: 2, pointMarkersVisible: true,
    });

    const sorted = [...pareto].sort((a, b) => a.cvar - b.cvar);
    const data = sorted.map((p, i) => ({ time: (i + 1) as any, value: p.expected_return * 365 * 100 }));
    series.setData(data);
  }
}
