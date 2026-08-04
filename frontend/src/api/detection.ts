import axios from 'axios';
import type { BlindDetectionResult, ComparativeResult, HistoryEntry } from '../types';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 60000,
});

export interface BlindDetectionOptions {
  confidenceThreshold: number; // 0-1
  enableGradcam: boolean;
  /** Averages over resample-free views; more accurate, ~4x slower. */
  useTta: boolean;
  /** Allow the system to answer UNCERTAIN instead of guessing. */
  allowAbstain: boolean;
}

const DEFAULT_OPTIONS: BlindDetectionOptions = {
  confidenceThreshold: 0.5,
  enableGradcam: true,
  useTta: true,
  allowAbstain: true,
};

function appendOptions(form: FormData, options: BlindDetectionOptions): void {
  form.append('confidence_threshold', String(options.confidenceThreshold));
  form.append('enable_gradcam', String(options.enableGradcam));
  form.append('use_tta', String(options.useTta));
  form.append('allow_abstain', String(options.allowAbstain));
}

export async function detectBlind(
  file: File,
  options: BlindDetectionOptions = DEFAULT_OPTIONS
): Promise<BlindDetectionResult> {
  const form = new FormData();
  form.append('image', file);
  appendOptions(form, options);
  const { data } = await api.post<BlindDetectionResult>('/detect/blind', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function downloadBlindReport(
  file: File,
  options: BlindDetectionOptions = DEFAULT_OPTIONS
): Promise<Blob> {
  const form = new FormData();
  form.append('image', file);
  appendOptions(form, options);
  const { data } = await api.post('/detect/blind/report', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    responseType: 'blob',
  });
  return data;
}

export async function getHistory(limit = 50): Promise<HistoryEntry[]> {
  const { data } = await api.get<HistoryEntry[]>('/detect/history', { params: { limit } });
  return data;
}

export async function clearHistory(): Promise<void> {
  await api.delete('/detect/history');
}

export async function detectComparative(
  original: File,
  suspected: File
): Promise<ComparativeResult> {
  const form = new FormData();
  form.append('original', original);
  form.append('suspected', suspected);
  const { data } = await api.post<ComparativeResult>('/detect/compare', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function downloadComparativeReport(original: File, suspected: File): Promise<Blob> {
  const form = new FormData();
  form.append('original', original);
  form.append('suspected', suspected);
  const { data } = await api.post('/detect/compare/report', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    responseType: 'blob',
  });
  return data;
}
