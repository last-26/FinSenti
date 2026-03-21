const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_PREFIX = "/api/v1";

async function fetchApi<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${API_PREFIX}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `API error: ${res.status}`);
  }
  return res.json();
}

// Types
export interface SentimentResult {
  text: string;
  sentiment: "positive" | "negative" | "neutral";
  confidence: number;
  probabilities: Record<string, number>;
  entities: string[];
  market_signal: string;
  model_used: string;
  inference_time_ms: number;
}

export interface BatchResponse {
  results: SentimentResult[];
  summary: {
    positive: number;
    negative: number;
    neutral: number;
    avg_confidence: number;
  };
  total_inference_time_ms: number;
}

export interface ModelInfo {
  name: string;
  base_model: string;
  is_active: boolean;
  status: string;
}

export interface ExperimentSummary {
  experiment_id: string;
  experiment_name: string;
  run_count: number;
}

export interface RunSummary {
  run_id: string;
  run_name: string;
  status: string;
  base_model: string | null;
  f1_macro: number | null;
  accuracy: number | null;
  latency_p50_ms: number | null;
}

export interface HistoryEntry {
  id: number;
  text: string;
  sentiment: string;
  confidence: number;
  model_used: string;
  created_at: string;
}

export interface HistoryResponse {
  entries: HistoryEntry[];
  total: number;
  page: number;
  page_size: number;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  model_name: string | null;
  device: string;
}

// API functions
export async function predict(text: string, model?: string): Promise<SentimentResult> {
  return fetchApi<SentimentResult>("/predict", {
    method: "POST",
    body: JSON.stringify({ text, model }),
  });
}

export async function predictBatch(texts: string[], model?: string): Promise<BatchResponse> {
  return fetchApi<BatchResponse>("/batch", {
    method: "POST",
    body: JSON.stringify({ texts, model }),
  });
}

export async function getModels(): Promise<ModelInfo[]> {
  return fetchApi<ModelInfo[]>("/models");
}

export async function getActiveModel(): Promise<ModelInfo> {
  return fetchApi<ModelInfo>("/models/active");
}

export async function switchModel(modelName: string): Promise<ModelInfo> {
  return fetchApi<ModelInfo>(`/models/switch?model_name=${modelName}`, {
    method: "POST",
  });
}

export async function getExperiments(): Promise<ExperimentSummary[]> {
  return fetchApi<ExperimentSummary[]>("/experiments");
}

export async function getExperimentRuns(experimentId: string): Promise<RunSummary[]> {
  return fetchApi<RunSummary[]>(`/experiments/${experimentId}/runs`);
}

export async function getHistory(page: number = 1, pageSize: number = 20): Promise<HistoryResponse> {
  return fetchApi<HistoryResponse>(`/history?page=${page}&page_size=${pageSize}`);
}

export async function getHealth(): Promise<HealthStatus> {
  return fetchApi<HealthStatus>("/health");
}
