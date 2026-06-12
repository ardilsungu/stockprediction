import {
  Component, OnInit, OnDestroy, ElementRef, ViewChild, signal, computed,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { AssetService } from '../../core/services/asset';
import { ForecastService } from '../../core/services/forecast';
import { Asset, ForecastJobDetail, PricePoint } from '../../core/models';
import {
  createChart, ColorType, LineSeries, LineStyle,
  IChartApi, ISeriesApi,
} from 'lightweight-charts';

@Component({
  selector: 'app-forecast',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './forecast.html',
  styleUrl: './forecast.scss',
})
export class Forecast implements OnInit, OnDestroy {
  @ViewChild('chartContainer') chartContainer!: ElementRef;

  form: FormGroup;
  submitting = false;
  errorMessage = '';
  pollInterval: any = null;
  private pollErrorCount = 0;
  private static readonly MAX_POLL_ERRORS = 3;

  currentJob = signal<ForecastJobDetail | null>(null);
  assets = signal<Asset[]>([]);

  isRunning = computed(() => {
    const status = this.currentJob()?.status;
    return status === 'pending' || status === 'running';
  });
  hasResult = computed(() => !!this.currentJob()?.result);

  private chart: IChartApi | null = null;
  private historySeries: ISeriesApi<'Line'> | null = null;
  private predictedSeries: ISeriesApi<'Line'> | null = null;
  private ciUpperSeries: ISeriesApi<'Line'> | null = null;
  private ciLowerSeries: ISeriesApi<'Line'> | null = null;
  private priceHistory: PricePoint[] = [];

  constructor(
    private fb: FormBuilder,
    private assetService: AssetService,
    private forecastService: ForecastService,
  ) {
    this.form = this.fb.group({
      asset:        ['', [Validators.required]],
      model_type:   ['prophet', [Validators.required]],
      horizon_days: [30, [Validators.required, Validators.min(1), Validators.max(365)]],
    });
  }

  ngOnInit(): void {
    this.assetService.getAssets().subscribe({
      next: (assets) => this.assets.set(assets),
      error: () => { this.errorMessage = 'Varlık listesi yüklenemedi.'; },
    });
  }

  ngOnDestroy(): void {
    this.clearPoll();
    this.disposeChart();
  }

  onSubmit(): void {
    if (this.form.invalid || this.submitting) return;
    this.submitting = true;
    this.errorMessage = '';
    this.currentJob.set(null);
    this.clearPoll();
    this.disposeChart();
    this.priceHistory = [];

    const { asset, model_type, horizon_days } = this.form.value;
    this.forecastService.createJob(asset, model_type, horizon_days).subscribe({
      next: (job) => {
        this.currentJob.set(job as ForecastJobDetail);
        this.submitting = false;
        this.startPolling(job.id);
      },
      error: () => {
        this.errorMessage = 'Tahmin işi başlatılamadı.';
        this.submitting = false;
      },
    });
  }

  private startPolling(jobId: string): void {
    this.pollErrorCount = 0;
    this.pollJob(jobId);
    this.pollInterval = setInterval(() => this.pollJob(jobId), 3000);
  }

  private pollJob(jobId: string): void {
    this.forecastService.getJobDetail(jobId).subscribe({
      next: (job) => {
        this.pollErrorCount = 0;
        this.currentJob.set(job);
        if (job.status === 'completed' || job.status === 'failed') {
          this.clearPoll();
          if (job.status === 'completed' && job.result) {
            this.loadHistoryAndRender(job.asset_symbol);
          }
        }
      },
      error: () => {
        // Token süresi dolduysa, job silindiyse veya backend düştüyse sonsuza
        // dek başarısız istek atmamak için ardışık hatada poll durdurulur.
        this.pollErrorCount++;
        if (this.pollErrorCount >= Forecast.MAX_POLL_ERRORS) {
          this.clearPoll();
          this.errorMessage = 'Job durumu alınamadı. Lütfen sayfayı yenileyip tekrar deneyin.';
        }
      },
    });
  }

  private loadHistoryAndRender(symbol: string): void {
    this.assetService.getPriceHistory(symbol, 30).subscribe({
      next: (history) => {
        this.priceHistory = history;
        setTimeout(() => this.renderChart(), 100);
      },
      error: () => {
        // Geçmiş veri alınamazsa grafiği sadece tahminlerle çiz
        this.priceHistory = [];
        setTimeout(() => this.renderChart(), 100);
      },
    });
  }

  private clearPoll(): void {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
  }

  private disposeChart(): void {
    this.historySeries = null;
    this.predictedSeries = null;
    this.ciUpperSeries = null;
    this.ciLowerSeries = null;
    if (this.chart) {
      this.chart.remove();
      this.chart = null;
    }
  }

  private renderChart(): void {
    const result = this.currentJob()?.result;
    if (!this.chartContainer?.nativeElement || !result || result.predictions.length === 0) return;
    this.disposeChart();

    this.chart = createChart(this.chartContainer.nativeElement, {
      width: this.chartContainer.nativeElement.offsetWidth,
      height: 400,
      layout: { background: { type: ColorType.Solid, color: '#0f1117' }, textColor: '#94a3b8' },
      grid:   { vertLines: { color: '#1e2235' }, horzLines: { color: '#1e2235' } },
      rightPriceScale: { borderColor: '#2a2d3e' },
      timeScale: { borderColor: '#2a2d3e' },
      crosshair: { mode: 1 },
    });

    // Geçmiş fiyatlar (gri çizgi)
    if (this.priceHistory.length > 0) {
      this.historySeries = this.chart.addSeries(LineSeries, {
        color: '#94a3b8',
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: false,
      });
      this.historySeries.setData(
        this.priceHistory.map(p => ({ time: p.date, value: p.close })),
      );
    }

    // Güven aralığı (kesikli açık çizgiler)
    const ciOptions = {
      color: 'rgba(99,102,241,0.4)',
      lineWidth: 1 as const,
      lineStyle: LineStyle.Dashed,
      priceLineVisible: false,
      lastValueVisible: false,
    };
    this.ciUpperSeries = this.chart.addSeries(LineSeries, ciOptions);
    this.ciUpperSeries.setData(
      result.predictions.map(p => ({ time: p.date, value: p.upper_ci })),
    );
    this.ciLowerSeries = this.chart.addSeries(LineSeries, ciOptions);
    this.ciLowerSeries.setData(
      result.predictions.map(p => ({ time: p.date, value: p.lower_ci })),
    );

    // Tahmin (indigo çizgi) — geçmişin son noktasından devam etsin
    this.predictedSeries = this.chart.addSeries(LineSeries, {
      color: '#6366f1',
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
    });
    const predData = result.predictions.map(p => ({ time: p.date, value: p.predicted }));
    const lastHist = this.priceHistory[this.priceHistory.length - 1];
    if (lastHist && lastHist.date < predData[0].time) {
      predData.unshift({ time: lastHist.date, value: lastHist.close });
    }
    this.predictedSeries.setData(predData);

    this.chart.timeScale().fitContent();
  }

  statusClass(status: string): string {
    return { pending: 'badge-pending', running: 'badge-running', completed: 'badge-completed', failed: 'badge-failed' }[status] ?? '';
  }

  statusLabel(status: string): string {
    return { pending: 'Bekliyor', running: 'Çalışıyor...', completed: 'Tamamlandı', failed: 'Hata' }[status] ?? status;
  }
}
