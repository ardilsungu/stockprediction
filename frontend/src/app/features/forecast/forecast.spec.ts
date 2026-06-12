import { FormBuilder } from '@angular/forms';
import { of, throwError } from 'rxjs';
import { delay } from 'rxjs/operators';
import { Forecast } from './forecast';

describe('Forecast — pollJob hata yönetimi', () => {
  const failure = () => throwError(() => new Error('HTTP 500'));

  function createComponent(getJobDetail: () => unknown): Forecast {
    const assetService = { getAssets: () => of([]), getPriceHistory: () => of([]) };
    const forecastService = { getJobDetail };
    return new Forecast(
      new FormBuilder(),
      assetService as never,
      forecastService as never,
    );
  }

  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('3 ardışık hatadan sonra poll durdurulur ve hata mesajı gösterilir', () => {
    const getJobDetail = vi.fn(failure);
    const component = createComponent(getJobDetail);

    (component as never as { startPolling(id: string): void }).startPolling('job-1');
    vi.advanceTimersByTime(6000); // ilk çağrı + 2 interval = 3 ardışık hata

    expect(component.errorMessage).toContain('Job durumu alınamadı');
    expect(component.pollInterval).toBeNull();

    // Poll durduğu için yeni istek atılmamalı
    vi.advanceTimersByTime(9000);
    expect(getJobDetail).toHaveBeenCalledTimes(3);
  });

  it('başarılı yanıt ardışık hata sayacını sıfırlar', () => {
    const runningJob = { id: 'job-1', status: 'running' };
    // 2 hata → 1 başarı → 2 hata: limit (3) hiç dolmaz, poll devam eder
    const responses = [failure(), failure(), of(runningJob), failure(), failure()];
    const getJobDetail = vi.fn(() => responses.shift() ?? of(runningJob));
    const component = createComponent(getJobDetail);

    (component as never as { startPolling(id: string): void }).startPolling('job-1');
    vi.advanceTimersByTime(12000);

    expect(component.errorMessage).toBe('');
    expect(component.pollInterval).not.toBeNull();
    component.ngOnDestroy();
  });

  it('completed yanıtı poll’u durdurur (mevcut davranış korunur)', () => {
    const completedJob = { id: 'job-1', status: 'completed', result: null, asset_symbol: 'BTC' };
    // delay(1): gerçek HTTP gibi asenkron yanıt
    const getJobDetail = vi.fn(() => of(completedJob).pipe(delay(1)));
    const component = createComponent(getJobDetail);

    (component as never as { startPolling(id: string): void }).startPolling('job-1');
    vi.advanceTimersByTime(1); // yanıt gelir → completed → clearPoll

    expect(component.pollInterval).toBeNull();
    vi.advanceTimersByTime(9000);
    expect(getJobDetail).toHaveBeenCalledTimes(1);
  });
});
