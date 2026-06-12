import { FormBuilder } from '@angular/forms';
import { of, throwError } from 'rxjs';
import { delay } from 'rxjs/operators';
import { Forecast } from './forecast';

const chartMock = vi.hoisted(() => {
  const series: Array<{ setData: ReturnType<typeof vi.fn> }> = [];
  const chart = {
    addSeries: vi.fn(() => {
      const s = { setData: vi.fn() };
      series.push(s);
      return s;
    }),
    timeScale: () => ({ fitContent: vi.fn() }),
    remove: vi.fn(),
  };
  return { series, chart, createChart: vi.fn(() => chart) };
});

vi.mock('lightweight-charts', () => ({
  createChart: chartMock.createChart,
  ColorType: { Solid: 'solid' },
  LineSeries: 'line',
  LineStyle: { Dashed: 2 },
}));

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

describe('Forecast — predictions.points ve baseline yapısı', () => {
  function createComponent(): Forecast {
    const assetService = { getAssets: () => of([]), getPriceHistory: () => of([]) };
    const forecastService = { getJobDetail: () => of(null) };
    return new Forecast(
      new FormBuilder(),
      assetService as never,
      forecastService as never,
    );
  }

  const result = {
    predictions: {
      points: [
        { date: '2026-06-14', predicted: 101, lower_ci: 93, upper_ci: 109 },
        { date: '2026-06-15', predicted: 102, lower_ci: 94, upper_ci: 110 },
      ],
      baseline: { mae: 3.5, rmse: 4.5 },
    },
    mae: 1.5,
    rmse: 2.5,
    created_at: '',
  };

  beforeEach(() => {
    chartMock.series.length = 0;
    chartMock.createChart.mockClear();
  });

  it('renderChart tahmin/CI serilerini predictions.points üzerinden besler', () => {
    const component = createComponent();
    component.currentJob.set({ id: 'job-1', status: 'completed', result } as never);
    (component as never as { chartContainer: unknown }).chartContainer = {
      nativeElement: { offsetWidth: 800 },
    };

    (component as never as { renderChart(): void }).renderChart();

    // priceHistory boş → sırasıyla 3 seri: CI üst, CI alt, tahmin
    expect(chartMock.series).toHaveLength(3);
    expect(chartMock.series[0].setData).toHaveBeenCalledWith([
      { time: '2026-06-14', value: 109 },
      { time: '2026-06-15', value: 110 },
    ]);
    expect(chartMock.series[2].setData).toHaveBeenCalledWith([
      { time: '2026-06-14', value: 101 },
      { time: '2026-06-15', value: 102 },
    ]);
  });

  it('points boşsa chart hiç oluşturulmaz', () => {
    const component = createComponent();
    const emptyResult = { ...result, predictions: { points: [], baseline: { mae: 0, rmse: 0 } } };
    component.currentJob.set({ id: 'job-1', status: 'completed', result: emptyResult } as never);
    (component as never as { chartContainer: unknown }).chartContainer = {
      nativeElement: { offsetWidth: 800 },
    };

    (component as never as { renderChart(): void }).renderChart();

    expect(chartMock.createChart).not.toHaveBeenCalled();
  });
});
