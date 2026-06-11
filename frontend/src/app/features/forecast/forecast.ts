import { Component, OnInit, OnDestroy, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { AssetService } from '../../core/services/asset';
import { ForecastService } from '../../core/services/forecast';
import { Asset, ForecastJobDetail } from '../../core/models';

@Component({
  selector: 'app-forecast',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './forecast.html',
  styleUrl: './forecast.scss',
})
export class Forecast implements OnInit, OnDestroy {
  form: FormGroup;
  submitting = false;
  errorMessage = '';
  pollInterval: any = null;

  currentJob = signal<ForecastJobDetail | null>(null);
  assets = signal<Asset[]>([]);

  isRunning = computed(() => {
    const status = this.currentJob()?.status;
    return status === 'pending' || status === 'running';
  });
  hasResult = computed(() => !!this.currentJob()?.result);

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
  }

  onSubmit(): void {
    if (this.form.invalid || this.submitting) return;
    this.submitting = true;
    this.errorMessage = '';
    this.currentJob.set(null);
    this.clearPoll();

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
    this.pollJob(jobId);
    this.pollInterval = setInterval(() => this.pollJob(jobId), 3000);
  }

  private pollJob(jobId: string): void {
    this.forecastService.getJobDetail(jobId).subscribe({
      next: (job) => {
        this.currentJob.set(job);
        if (job.status === 'completed' || job.status === 'failed') {
          this.clearPoll();
        }
      },
    });
  }

  private clearPoll(): void {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
  }

  statusClass(status: string): string {
    return { pending: 'badge-pending', running: 'badge-running', completed: 'badge-completed', failed: 'badge-failed' }[status] ?? '';
  }

  statusLabel(status: string): string {
    return { pending: 'Bekliyor', running: 'Çalışıyor...', completed: 'Tamamlandı', failed: 'Hata' }[status] ?? status;
  }
}
