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

/**
 * Provenance / container forensics — the Tier A evidence stream.
 *
 * `absence_is_not_evidence` is always true and is carried through from the
 * backend deliberately: the UI must never present "no manifest found" or
 * "no camera EXIF" as evidence the image is AI-generated. Most real photos
 * carry neither — every screenshot and social-media upload strips them.
 */
export interface ProvenanceFindings {
  c2pa_manifest_present: boolean;
  c2pa_verified: boolean;
  has_exif: boolean;
  camera_evidence_strength: 'none' | 'weak' | 'moderate' | 'strong';
  software_tags: string[];
  generative_software_hint: string | null;
  jpeg_quantization_fingerprint: string | null;
  quantization_notes: string[];
  thumbnail_mismatch: boolean | null;
  tier_a_resolved: boolean;
  reasons: string[];
  absence_is_not_evidence: boolean;
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
  /**
   * Which evidence tier produced this answer — see
   * PLAN_HIGH_CONFIDENCE_DETECTION.md §3. A_cryptographic is near-certain
   * (verified signed provenance, no model involved); B_forensic is model +
   * heuristic consensus; C_uncertain means the system declined to commit.
   * Optional: responses from an older backend won't carry it.
   */
  evidence_tier?: 'A_cryptographic' | 'B_forensic' | 'C_uncertain';
  provenance?: ProvenanceFindings;
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
