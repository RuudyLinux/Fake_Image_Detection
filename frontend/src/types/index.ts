export type PredictionClass = 'REAL' | 'AI_GENERATED' | 'EDITED';

export interface BlindDetectionResult {
  prediction: PredictionClass;
  confidence: number;
  forgery_type: string;
  heatmap_url: string | null;
  class_probabilities: {
    REAL: number;
    AI_GENERATED: number;
    EDITED: number;
  };
  signal_agreement: boolean | null;
  analysis: {
    edge_score:          number;
    fft_score:           number;
    gan_score:           number;
    noise_score:         number;
    texture_score:       number;
    heuristic_composite: number;
  };
}

export interface ComparativeResult {
  similarity_score: number;
  is_manipulated: boolean;
  diff_map_url: string | null;
  highlighted_url: string | null;
  regions_count: number;
}

export type DetectionStatus = 'idle' | 'uploading' | 'analyzing' | 'done' | 'error';

export interface HistoryEntry {
  id: number;
  created_at: string;
  mode: 'blind' | 'compare';
  prediction: string | null;
  confidence: number | null;
  forgery_type: string | null;
  thumbnail_url: string | null;
  heatmap_url: string | null;
}
