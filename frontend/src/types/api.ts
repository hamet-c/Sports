export type StatType =
  | 'points'
  | 'rebounds'
  | 'assists'
  | 'threes_made'
  | 'pra';

export interface SlateProp {
  player_id: number;
  player_name: string;
  game_id: number;
  game_date: string;
  matchup: string;
  team_abbr: string;
  opp_abbr: string;
  home_abbr: string;
  away_abbr: string;
  is_home: boolean;
  stat_type: StatType | string;
  line: number;
  over_odds: number;
  under_odds: number;
  book: string;
  predicted_mean: number;
  over_probability: number;
  expected_value_over: number;
  expected_value_under: number;
  kelly_over: number;
  kelly_under: number;
  recommendation: 'OVER' | 'UNDER' | 'PASS';
  sharp_book_disagreement: boolean;
  book_favored_side: 'OVER' | 'UNDER' | 'EVEN';
}

export interface SlateResponse {
  date: string;
  props: SlateProp[];
}

export interface Player {
  id: number;
  full_name: string;
  position: string | null;
  team_id: number | null;
}

export interface PlayerGameStat {
  game_date: string;
  minutes: number | null;
  points: number | null;
  rebounds: number | null;
  assists: number | null;
  threes_made: number | null;
  is_home: boolean | null;
}

export interface StatPrediction {
  stat_type: string;
  predicted_mean: number;
  quantiles: Record<string, number>;
}

export interface PredictionResponse {
  player_id: number;
  game_id: number;
  as_of: string;
  predictions: StatPrediction[];
}

export interface CalibrationBucket {
  n: number;
  hit_rate: number;
}

export interface RecommendationSideRecord {
  n: number;
  wins: number;
  win_rate: number | null;
}

export interface RecommendationsRecord {
  n: number;
  pushes: number;
  wins: number;
  win_rate: number | null;
  over: RecommendationSideRecord;
  under: RecommendationSideRecord;
}

export interface SyntheticStatReport {
  n: number;
  mae: number;
  rmse: number;
  log_loss: number;
  baseline_l10_mae: number | null;
  vs_baseline_lift: number | null;
  calibration: Record<string, CalibrationBucket>;
  recommendations?: RecommendationsRecord;
}

export interface SyntheticReportEnvelope {
  mode: 'synthetic';
  generated_at: string;
  start: string;
  end: string;
  result: Record<string, SyntheticStatReport>;
}

export interface RealStatReport {
  n_predictions: number;
  n_bets: number;
  n_wins: number;
  win_rate: number;
  total_staked: number;
  total_returned: number;
  roi: number;
  log_loss: number;
}

export interface RealReportEnvelope {
  mode: 'real';
  generated_at: string;
  start: string;
  end: string;
  result: RealStatReport & {
    by_stat: Record<string, RealStatReport>;
    calibration: Record<string, { n: number; hits: number; hit_rate: number }>;
  };
}

export interface PerformanceResponse {
  synthetic: SyntheticReportEnvelope | null;
  real: RealReportEnvelope | null;
}

export interface StatComparison {
  actual: number | null;
  predicted: number;
  error: number | null;
}

export interface RecentPredictionComparison {
  game_id: number;
  game_date: string;
  opponent_abbr: string | null;
  is_home: boolean | null;
  minutes: number | null;
  stats: Record<string, StatComparison>;
}

export interface FeatureImportanceEntry {
  feature: string;
  importance: number;
}

export interface FeatureImportanceResponse {
  stats: Record<string, FeatureImportanceEntry[]>;
}

export interface CoverageResponse {
  generated_at: string;
  train_end: string;
  val_end: string;
  feature_columns: string[];
  coverage: Record<string, Record<string, number>>;
}

export interface RecRecordSide {
  n: number;
  wins: number;
  losses: number;
  pushes: number;
  win_rate: number | null;
}

export interface RecRecordByStat {
  n: number;
  wins: number;
  losses: number;
  pushes: number;
  win_rate: number | null;
}

export interface RecRecordResponse {
  start: string;
  end: string;
  n_recommendations: number;
  wins: number;
  losses: number;
  pushes: number;
  win_rate: number | null;
  by_stat: Record<string, RecRecordByStat>;
  over: RecRecordSide;
  under: RecRecordSide;
  note: string;
}

export interface SlateAnchorResponse {
  today: string;
  latest_prop_date: string | null;
  days_stale: number | null;
}
