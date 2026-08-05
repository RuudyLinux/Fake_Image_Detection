export type PredictionClass = 'REAL' | 'AI_GENERATED' | 'EDITED';
export type Verdict = PredictionClass | 'UNCERTAIN';

export interface UncertaintyBreakdown {
  /** Total uncertainty in the fused distribution (nats). */
  predictive_entropy: number;
  /** Irreducible ambiguity — the image genuinely sits near a class boundary. */
  aleatoric: number;
  /** Model disagreement — high values flag inputs unlike the training data. */
  epistemic: number;
}

export interface EnsembleMeta {
  members: string[];
  members_agree: boolean;
  tta_used: boolean;
  temperature: number;
}

export interface BlindDetectionResult {
  /** UNCERTAIN when the system declines to commit. */
  prediction: Verdict;
  /** What the models leaned toward, even when the verdict is UNCERTAIN. */
  model_prediction: PredictionClass;
  confidence: number;
  meets_threshold: boolean;
  abstained: boolean;
  abstain_reasons: string[];
  forgery_type: string;
  heatmap_url: string | null;
  class_probabilities: {
    REAL: number;
    AI_GENERATED: number;
    EDITED: number;
  };
  signal_agreement: boolean | null;
  uncertainty: UncertaintyBreakdown | null;
  member_probabilities: Record<string, Record<string, number>> | null;
  ensemble: EnsembleMeta | null;
  analysis: {
    edge_score:          number;
    fft_score:           number;
    gan_score:           number;
    noise_score:         number;
    texture_score:       number;
    ela_score:           number;
    heuristic_composite: number;
  };
}

export interface ComparativeEvidence {
  /** Perceptual-hash distance indicates a real content change. */
  hash_evidence: boolean;
  /** Differences are concentrated in regions rather than spread diffusely. */
  localized_evidence: boolean;
  /** Human-readable justification, including the negative case. */
  reasons: string[];
}

export interface ComparativeResult {
  similarity_score: number;
  is_manipulated: boolean;
  diff_map_url: string | null;
  highlighted_url: string | null;
  regions_count: number;
  phash_distance: number;
  dhash_distance: number;
  /**
   * Mean pixel difference inside the flagged regions over the mean outside.
   * ~1 means the difference is spread evenly (re-compression / resizing);
   * large values mean it is confined to specific regions (local editing).
   * This is what the verdict keys on — SSIM magnitude alone cannot separate
   * the two cases. Optional so an older backend still type-checks.
   */
  localization_ratio?: number;
  evidence?: ComparativeEvidence;
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
