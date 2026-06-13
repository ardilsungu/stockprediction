import { TestBed } from '@angular/core/testing';
import { FormBuilder } from '@angular/forms';
import { of } from 'rxjs';
import { Portfolio } from './portfolio';
import {
  ParetoSolution, PortfolioJobDetail, StrategyMetrics,
} from '../../core/models';

// lightweight-charts modülü test ortamında çözülebilsin (grafik kullanılmıyor).
vi.mock('lightweight-charts', () => ({
  createChart: vi.fn(),
  ColorType: { Solid: 'solid' },
  LineSeries: 'line',
  createSeriesMarkers: vi.fn(),
}));

function makeStrategy(over: Partial<StrategyMetrics> = {}): StrategyMetrics {
  return {
    name: 'X', annual_return: 0.1, annual_vol: 0.2, sharpe: 0.5, sortino: 0.6,
    calmar: 0.3, omega: 1.2, cvar_daily: 0.02, cvar_annual: 0.3,
    max_drawdown: 0.25, n_active: 8, is_duplicate: false, ...over,
  };
}

function createComponent(): Portfolio {
  const portfolioService = { getJobDetail: vi.fn(), createJob: vi.fn() };
  const route = { queryParams: of({}) };
  // Portfolio constructor'ı effect() kullandığı için injection context gerekir.
  return TestBed.runInInjectionContext(
    () => new Portfolio(new FormBuilder(), portfolioService as never, route as never),
  );
}

describe('Portfolio — out-of-sample (holdout) metadata', () => {
  it('evaluation() params.evaluation değerini döndürür', () => {
    const component = createComponent();
    component.currentJob.set({
      id: 'j1', status: 'completed',
      params: {
        evaluation: {
          method: 'holdout', test_fraction: 0.2, n_train: 160, n_test: 40,
          test_start: '2024-05-01', test_end: '2024-06-10',
        },
      },
    } as unknown as PortfolioJobDetail);

    const ev = component.evaluation();
    expect(ev?.method).toBe('holdout');
    expect(ev?.n_test).toBe(40);
    expect(ev?.test_start).toBe('2024-05-01');
  });

  it('evaluation() metadata yoksa null döner', () => {
    const component = createComponent();
    component.currentJob.set({
      id: 'j1', status: 'completed', params: {},
    } as unknown as PortfolioJobDetail);
    expect(component.evaluation()).toBeNull();
  });

  it('strategies() pareto_index dahil metrikleri map eder', () => {
    const component = createComponent();
    component.currentJob.set({
      id: 'j1', status: 'completed', params: {},
      result: {
        id: 'r1', surviving_assets: [], pareto_solutions: [], weights: {},
        created_at: '', strategies: {
          max_sharpe: makeStrategy({ name: 'Max Sharpe', pareto_index: 2 }),
        },
      },
    } as unknown as PortfolioJobDetail);

    const rows = component.strategies();
    expect(rows).toHaveLength(1);
    expect(rows[0].key).toBe('max_sharpe');
    expect(rows[0].pareto_index).toBe(2);
  });
});

describe('Portfolio — grafik işaretçileri pareto_index ile konumlanır', () => {
  it('strateji, metrik mesafesiyle değil pareto_index ile doğru noktaya bağlanır', () => {
    const component = createComponent();
    // Orijinal sıra: idx0 cvar=0.3, idx1 cvar=0.1, idx2 cvar=0.2
    const pareto: ParetoSolution[] = [
      { neg_return: -0.1, cvar: 0.3, expected_return: 0.1 },
      { neg_return: -0.2, cvar: 0.1, expected_return: 0.2 },
      { neg_return: -0.3, cvar: 0.2, expected_return: 0.3 },
    ];
    // pareto_index=0 → orijinal 0. numaralı çözüm (cvar 0.3) işaretlenmeli.
    // annual_return/cvar_annual kasıtlı olarak alakasız (OOS) verildi ki
    // mesafe eşlemesi DEĞİL pareto_index'in kullanıldığı görülsün.
    const strategies: Record<string, StrategyMetrics> = {
      max_sharpe: makeStrategy({
        pareto_index: 0, annual_return: 999, cvar_annual: 999,
      }),
    };

    const points = (component as never as {
      buildPlotPoints(p: ParetoSolution[], s: Record<string, StrategyMetrics>): Array<{ origIndex: number; strategyKey?: string }>;
    }).buildPlotPoints(pareto, strategies);

    const marked = points.find(p => p.strategyKey === 'max_sharpe');
    expect(marked).toBeDefined();
    expect(marked!.origIndex).toBe(0);
  });
});
