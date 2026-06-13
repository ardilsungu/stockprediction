import {
  Component, OnInit, OnDestroy, AfterViewInit,
  ElementRef, ViewChild, signal, computed, effect,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute } from '@angular/router';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { PortfolioService } from '../../core/services/portfolio';
import {
  ParetoSolution, PortfolioEvaluation, PortfolioJobDetail, StrategyMetrics,
} from '../../core/models';
import {
  createChart, ColorType, LineSeries,
  createSeriesMarkers, ISeriesApi, IChartApi, ISeriesMarkersPluginApi,
  MouseEventParams, SeriesMarker, Time,
} from 'lightweight-charts';

const STRATEGY_COLORS: Record<string, string> = {
  max_sharpe:  '#eab308',
  max_sortino: '#22c55e',
  min_cvar:    '#3b82f6',
  max_return:  '#ef4444',
  balanced:    '#f97316',
};

const STRATEGY_LABELS: Record<string, string> = {
  max_sharpe:  'Max Sharpe',
  max_sortino: 'Max Sortino',
  min_cvar:    'Min CVaR',
  max_return:  'Max Return',
  balanced:    'Balanced',
};

const STRATEGY_LEGEND = Object.keys(STRATEGY_LABELS).map(key => ({
  key,
  label: STRATEGY_LABELS[key],
  color: STRATEGY_COLORS[key],
}));

interface StrategyRow extends StrategyMetrics {
  key: string;
}

interface WeightRow {
  ticker: string;
  weight: number;
}

interface PlotPoint {
  index: number;
  annualCvar: number;
  annualReturn: number;
  origIndex: number;   // sıralamadan önceki pareto indeksi (backend pareto_index)
  strategyKey?: string;
}

interface TooltipState {
  x: number;
  y: number;
  annualCvar: number;
  annualReturn: number;
  strategyKey?: string;
  strategyName?: string;
  sharpe?: number;
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
  tooltip = signal<TooltipState | null>(null);

  readonly strategyLegend = STRATEGY_LEGEND;
  readonly strategyColors = STRATEGY_COLORS;

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

  // Metriklerin hesaplandığı out-of-sample (holdout) pencere bilgisi (varsa).
  evaluation = computed<PortfolioEvaluation | null>(
    () => this.currentJob()?.params?.evaluation ?? null,
  );

  pollInterval: any = null;
  errorMessage = '';
  private chart: IChartApi | null = null;
  private mainSeries: ISeriesApi<'Line'> | null = null;
  private markersApi: ISeriesMarkersPluginApi<Time> | null = null;
  private plotPoints: PlotPoint[] = [];

  constructor(
    private fb: FormBuilder,
    private portfolioService: PortfolioService,
    private route: ActivatedRoute,
  ) {
    this.form = this.fb.group({
      n_coins:      [50,  [Validators.required, Validators.min(20),  Validators.max(100)]],
      lookback_days:[365, [Validators.required, Validators.min(250), Validators.max(750)]],
      n_gen:        [100, [Validators.required, Validators.min(20),  Validators.max(500)]],
      pop_size:     [100, [Validators.required, Validators.min(20),  Validators.max(500)]],
    });

    effect(() => {
      const sel = this.selectedStrategy();
      this.updateMarkers(sel);
    });
  }

  ngOnInit(): void {
    this.route.queryParams.subscribe((params) => {
      const jobId = params['jobId'];
      if (jobId) this.loadJobById(jobId);
    });
  }
  ngAfterViewInit(): void {}

  private loadJobById(jobId: string): void {
    this.clearPoll();
    this.disposeChart();
    this.currentJob.set(null);
    this.selectedStrategy.set(null);
    this.tooltip.set(null);
    this.errorMessage = '';
    this.startPolling(jobId);
  }

  ngOnDestroy(): void {
    this.clearPoll();
    this.disposeChart();
  }

  startJob(): void {
    if (this.form.invalid || this.submitting) return;
    this.submitting = true;
    this.currentJob.set(null);
    this.selectedStrategy.set(null);
    this.tooltip.set(null);
    this.errorMessage = '';
    this.disposeChart();

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
            setTimeout(
              () => this.renderChart(job.result?.pareto_solutions ?? [], job.result?.strategies ?? {}),
              100,
            );
          }
        }
      },
    });
  }

  private clearPoll(): void {
    if (this.pollInterval) { clearInterval(this.pollInterval); this.pollInterval = null; }
  }

  private disposeChart(): void {
    this.markersApi = null;
    this.mainSeries = null;
    this.plotPoints = [];
    this.tooltip.set(null);
    if (this.chart) {
      this.chart.remove();
      this.chart = null;
    }
  }

  private renderChart(pareto: ParetoSolution[], strategies: Record<string, StrategyMetrics>): void {
    if (!this.chartContainer?.nativeElement || pareto.length === 0) return;
    this.disposeChart();

    this.chart = createChart(this.chartContainer.nativeElement, {
      width: this.chartContainer.nativeElement.offsetWidth,
      height: 400,
      layout: { background: { type: ColorType.Solid, color: '#0f1117' }, textColor: '#94a3b8' },
      grid:   { vertLines: { color: '#1e2235' }, horzLines: { color: '#1e2235' } },
      rightPriceScale: { borderColor: '#2a2d3e' },
      timeScale: {
        borderColor: '#2a2d3e',
        visible: true,
        tickMarkFormatter: (time: Time) => {
          const pt = this.plotPoints.find(p => p.index === Number(time));
          if (!pt) return '';
          return `${(pt.annualCvar * 100).toFixed(1)}%`;
        },
      },
      crosshair: { mode: 1 },
      localization: {
        priceFormatter: (price: number) => `${price.toFixed(2)}%`,
      },
    });

    this.mainSeries = this.chart.addSeries(LineSeries, {
      color: '#475569',
      lineWidth: 2,
      pointMarkersVisible: true,
      pointMarkersRadius: 4,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    this.plotPoints = this.buildPlotPoints(pareto, strategies);

    this.mainSeries.setData(
      this.plotPoints.map(p => ({
        time: p.index as unknown as Time,
        value: p.annualReturn * 100,
      }))
    );

    this.markersApi = createSeriesMarkers<Time>(this.mainSeries, this.buildMarkers(this.selectedStrategy()));
    this.chart.timeScale().fitContent();

    this.chart.subscribeClick((p) => this.onChartClick(p));
    this.chart.subscribeCrosshairMove((p) => this.onCrosshair(p));
  }

  private buildPlotPoints(
    pareto: ParetoSolution[],
    strategies: Record<string, StrategyMetrics>,
  ): PlotPoint[] {
    const SQRT_365 = Math.sqrt(365);
    // Orijinal pareto indeksini sıralama boyunca koru — backend'in pareto_index'i
    // buna işaret eder.
    const sorted = pareto
      .map((p, origIndex) => ({ ...p, origIndex }))
      .sort((a, b) => a.cvar - b.cvar);
    const points: PlotPoint[] = sorted.map((p, i) => ({
      index: i + 1,
      annualCvar: p.cvar * SQRT_365,
      annualReturn: p.expected_return * 365,
      origIndex: p.origIndex,
    }));

    // İşaretçileri pareto_index ile yerleştir: metrikler artık out-of-sample
    // olduğu için in-sample cephe koordinatlarıyla mesafe eşlemesi kayardı.
    // Eski sonuçlarda (pareto_index yok) güvenli geri-dönüş: metrik mesafesi.
    for (const [key, s] of Object.entries(strategies)) {
      if (s.pareto_index != null) {
        const pt = points.find(p => p.origIndex === s.pareto_index && !p.strategyKey);
        if (pt) { pt.strategyKey = key; continue; }
      }
      let bestIdx = -1;
      let bestDist = Infinity;
      for (let i = 0; i < points.length; i++) {
        if (points[i].strategyKey) continue;
        const dr = points[i].annualReturn - s.annual_return;
        const dc = points[i].annualCvar  - s.cvar_annual;
        const d  = dr * dr + dc * dc;
        if (d < bestDist) { bestDist = d; bestIdx = i; }
      }
      if (bestIdx >= 0) points[bestIdx].strategyKey = key;
    }

    return points;
  }

  private buildMarkers(selectedKey: string | null): SeriesMarker<Time>[] {
    return this.plotPoints
      .filter(p => p.strategyKey)
      .map(p => {
        const key = p.strategyKey!;
        const isSelected = key === selectedKey;
        return {
          time: p.index as unknown as Time,
          position: 'aboveBar',
          color: STRATEGY_COLORS[key] ?? '#94a3b8',
          shape: 'circle',
          text: STRATEGY_LABELS[key] ?? key,
          size: isSelected ? 2.5 : 1.5,
        } as SeriesMarker<Time>;
      });
  }

  private updateMarkers(selectedKey: string | null): void {
    if (!this.markersApi) return;
    this.markersApi.setMarkers(this.buildMarkers(selectedKey));
  }

  private onChartClick(param: MouseEventParams<Time>): void {
    if (param.time == null) return;
    const t = Number(param.time);
    const pt = this.plotPoints.find(p => p.index === t);
    if (!pt) return;
    if (pt.strategyKey) {
      this.selectedStrategy.set(pt.strategyKey);
      return;
    }

    // Strateji-dışı bir noktaya tıklanınca grafik-uzayında en yakın strateji
    // işaretçisine snap et (metrik değil konum mesafesi — OOS metriklerden bağımsız).
    let bestKey: string | null = null;
    let bestDist = Infinity;
    for (const p of this.plotPoints) {
      if (!p.strategyKey) continue;
      const dr = pt.annualReturn - p.annualReturn;
      const dc = pt.annualCvar  - p.annualCvar;
      const d  = dr * dr + dc * dc;
      if (d < bestDist) { bestDist = d; bestKey = p.strategyKey; }
    }
    if (bestKey) this.selectedStrategy.set(bestKey);
  }

  private onCrosshair(param: MouseEventParams<Time>): void {
    if (!param.point || param.time == null || !this.mainSeries || !param.seriesData?.has(this.mainSeries)) {
      this.tooltip.set(null);
      return;
    }
    const t = Number(param.time);
    const pt = this.plotPoints.find(p => p.index === t);
    if (!pt) { this.tooltip.set(null); return; }
    const strategiesDict = this.currentJob()?.result?.strategies;
    const sharpe = pt.strategyKey ? strategiesDict?.[pt.strategyKey]?.sharpe : undefined;
    this.tooltip.set({
      x: param.point.x,
      y: param.point.y,
      annualCvar:   pt.annualCvar,
      annualReturn: pt.annualReturn,
      strategyKey:  pt.strategyKey,
      strategyName: pt.strategyKey ? (STRATEGY_LABELS[pt.strategyKey] ?? pt.strategyKey) : undefined,
      sharpe:       typeof sharpe === 'number' ? sharpe : undefined,
    });
  }
}
