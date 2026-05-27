import axios from 'axios';
import type {
  CoverageResponse,
  FeatureImportanceResponse,
  PerformanceResponse,
  Player,
  PlayerGameStat,
  PredictionResponse,
  RecRecordResponse,
  RecentPredictionComparison,
  SlateAnchorResponse,
  SlateResponse,
} from '../types/api';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? '/api/v1',
  timeout: 20_000,
});

export async function fetchSlate(
  date?: string,
  minEdge = 0,
  book?: string,
): Promise<SlateResponse> {
  const { data } = await api.get<SlateResponse>('/slate/', {
    params: { target_date: date, min_edge: minEdge, book },
  });
  return data;
}

export async function fetchRecommendationRecord(
  days = 7,
): Promise<RecRecordResponse> {
  const { data } = await api.get<RecRecordResponse>('/slate/recommendation_record', {
    params: { days },
  });
  return data;
}

export async function fetchSlateAnchor(): Promise<SlateAnchorResponse> {
  const { data } = await api.get<SlateAnchorResponse>('/slate/anchor');
  return data;
}

export async function searchPlayers(query: string): Promise<Player[]> {
  const { data } = await api.get<Player[]>('/players/', {
    params: { name_contains: query, limit: 20 },
  });
  return data;
}

export async function getPlayer(playerId: number): Promise<Player> {
  const { data } = await api.get<Player>(`/players/${playerId}`);
  return data;
}

export async function getPlayerRecent(
  playerId: number,
  limit = 15,
): Promise<PlayerGameStat[]> {
  const { data } = await api.get<PlayerGameStat[]>(`/players/${playerId}/recent`, {
    params: { limit },
  });
  return data;
}

export async function fetchPredictionsVsActual(
  playerId: number,
  limit = 3,
): Promise<RecentPredictionComparison[]> {
  const { data } = await api.get<RecentPredictionComparison[]>(
    `/players/${playerId}/predictions_vs_actual`,
    { params: { limit } },
  );
  return data;
}

export async function predictForGame(
  playerId: number,
  gameId: number,
): Promise<PredictionResponse> {
  const { data } = await api.post<PredictionResponse>('/predictions/', {
    player_id: playerId,
    game_id: gameId,
  });
  return data;
}

export interface HealthResponse {
  status: string;
  models_loaded: number;
  stats: string[];
}

export async function fetchPerformance(): Promise<PerformanceResponse> {
  const { data } = await api.get<PerformanceResponse>('/performance/');
  return data;
}

export async function fetchFeatureImportance(
  topK = 15,
): Promise<FeatureImportanceResponse> {
  const { data } = await api.get<FeatureImportanceResponse>(
    '/performance/feature_importance',
    { params: { top_k: topK } },
  );
  return data;
}

export async function fetchCoverage(): Promise<CoverageResponse> {
  const { data } = await api.get<CoverageResponse>('/performance/coverage');
  return data;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await axios.get<HealthResponse>('/health', {
    baseURL: (import.meta.env.VITE_API_URL ?? '/api/v1').replace('/api/v1', ''),
    timeout: 5_000,
  });
  return data;
}
