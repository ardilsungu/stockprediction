export interface User {
  id: string;
  email: string;
  username: string;
  created_at: string;
}

export interface LoginResponse {
  access: string;
  refresh: string;
}

export interface RegisterResponse {
  message: string;
  access: string;
  refresh: string;
}

export interface LogoutResponse {
  message: string;
}

export interface Asset {
  id: string;
  symbol: string;
  name: string;
  market_cap: number | null;
  category: string | null;
  is_active: boolean;
}

export interface WatchlistItem {
  id: string;
  asset: Asset;
  added_at: string;
}

export type PortfolioJobStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface PortfolioJobParams {
  lookback_days?: number;
  n_coins?: number;
  n_gen?: number;
  pop_size?: number;
  max_weight?: number;
  min_assets?: number;
  max_assets?: number;
  n_obj?: number;
  [key: string]: unknown;
}

export interface PortfolioJob {
  id: string;
  status: PortfolioJobStatus;
  params: PortfolioJobParams;
  celery_task_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface StrategyMetrics {
  name: string;
  annual_return: number;
  annual_vol: number;
  sharpe: number;
  sortino: number;
  calmar: number;
  omega: number;
  cvar_daily: number;
  cvar_annual: number;
  max_drawdown: number;
  n_active: number;
  is_duplicate?: boolean;
}

export interface ParetoSolution {
  neg_return: number;
  cvar: number;
  expected_return: number;
}

export interface PortfolioResult {
  id: string;
  surviving_assets: string[];
  pareto_solutions: ParetoSolution[];
  strategies: Record<string, StrategyMetrics>;
  weights: Record<string, Record<string, number>>;
  created_at: string;
}

export interface PortfolioJobDetail extends PortfolioJob {
  result?: PortfolioResult;
}
