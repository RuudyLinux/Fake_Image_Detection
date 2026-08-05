import { useState } from 'react';
import { GitCompare, Loader2, AlertCircle, RotateCcw, ArrowDown } from 'lucide-react';
import UploadZone from '../components/UploadZone';
import ComparativeResultPanel from '../components/ComparativeResultPanel';
import Reveal from '../components/Reveal';
import { detectComparative } from '../api/detection';
import type { ComparativeResult, DetectionStatus } from '../types';

const TECHNIQUES = [
  { name: 'Perceptual hashing', desc: 'pHash / dHash distance — survives resize and re-compression' },
  { name: 'Structural similarity', desc: 'SSIM map over the aligned pair' },
  { name: 'Localization ratio', desc: 'Separates local edits from diffuse compression noise' },
  { name: 'Region contouring', desc: 'Bounds and highlights each altered area' },
];

export default function ComparativeDetection() {
  const [original, setOriginal] = useState<File | null>(null);
  const [suspected, setSuspected] = useState<File | null>(null);
  const [status, setStatus] = useState<DetectionStatus>('idle');
  const [result, setResult] = useState<ComparativeResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const analysing = status === 'analyzing';
  const canAnalyze = !!original && !!suspected && !analysing;

  const handleAnalyze = async () => {
    if (!original || !suspected) return;
    setStatus('analyzing');
    setError(null);
    setResult(null);
    try {
      setResult(await detectComparative(original, suspected));
      setStatus('done');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Comparison failed. Is the backend running?');
      setStatus('error');
    }
  };

  const handleReset = () => {
    setOriginal(null);
    setSuspected(null);
    setResult(null);
    setError(null);
    setStatus('idle');
  };

  return (
    <div className="px-6 pb-24 pt-28">
      <div className="mx-auto max-w-6xl">
        <div className="mb-10 animate-fade-up">
          <span className="eyebrow">
            <GitCompare size={12} /> Comparative detection
          </span>
          <h1 className="mt-4 font-display text-4xl font-bold tracking-tight sm:text-5xl">
            Compare with the <span className="gradient-text">original</span>
          </h1>
          <p className="mt-3 max-w-xl text-slate-400">
            Upload both images. Localization-ratio analysis distinguishes a genuine edit
            from harmless re-compression, then pinpoints exactly what changed.
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-5">
          {/* ---------- inputs ---------- */}
          <div className="space-y-5 lg:col-span-2">
            <Reveal>
              <div className="card card-lit space-y-4">
                <UploadZone
                  label="Original image"
                  file={original}
                  onFileChange={setOriginal}
                  scanning={analysing}
                />

                <div className="flex items-center gap-3">
                  <span className="h-px flex-1 bg-gradient-to-r from-transparent to-white/10" />
                  <span className="flex h-7 w-7 items-center justify-center rounded-full border border-white/10 bg-white/[.03]">
                    <ArrowDown size={13} className="text-slate-500" />
                  </span>
                  <span className="h-px flex-1 bg-gradient-to-l from-transparent to-white/10" />
                </div>

                <UploadZone
                  label="Suspected image"
                  file={suspected}
                  onFileChange={setSuspected}
                  scanning={analysing}
                  accent="cyan"
                />

                <button onClick={handleAnalyze} disabled={!canAnalyze} className="btn-primary w-full">
                  {analysing ? (
                    <><Loader2 size={16} className="animate-spin" /> Comparing…</>
                  ) : (
                    <><GitCompare size={16} /> Compare images</>
                  )}
                </button>

                {(result || error) && (
                  <button onClick={handleReset} className="btn-secondary w-full text-sm">
                    <RotateCcw size={15} /> Reset
                  </button>
                )}
              </div>
            </Reveal>

            <Reveal delay={100}>
              <div className="card space-y-3.5">
                <p className="label-mono">Techniques</p>
                {TECHNIQUES.map(({ name, desc }, i) => (
                  <div key={name} className="flex gap-3">
                    <span className="mt-0.5 font-mono text-[10px] text-slate-600">0{i + 1}</span>
                    <div>
                      <p className="text-sm font-medium text-slate-300">{name}</p>
                      <p className="mt-0.5 font-mono text-[10px] leading-relaxed text-slate-600">{desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </Reveal>
          </div>

          {/* ---------- results ---------- */}
          <div className="lg:col-span-3">
            {status === 'idle' && (
              <Reveal delay={150}>
                <div className="card flex min-h-[22rem] flex-col items-center justify-center gap-4 border-dashed text-center">
                  <span className="flex h-16 w-16 items-center justify-center rounded-2xl border border-white/10 bg-white/[.03]">
                    <GitCompare size={26} className="text-slate-600" />
                  </span>
                  <div>
                    <p className="font-medium text-slate-300">Awaiting both images</p>
                    <p className="mt-1 font-mono text-[11px] text-slate-600">
                      Upload an original and a suspected copy
                    </p>
                  </div>
                </div>
              </Reveal>
            )}

            {analysing && (
              <div className="card card-lit scanlines flex min-h-[22rem] flex-col items-center justify-center gap-5 overflow-hidden">
                <div className="relative flex h-20 w-20 items-center justify-center">
                  <span className="absolute inset-0 rounded-full border border-cyan-500/40 animate-pulse-ring" />
                  <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-500 to-violet-600">
                    <GitCompare size={24} className="text-white" />
                  </span>
                </div>
                <div className="text-center">
                  <p className="font-medium text-slate-200">Comparing images…</p>
                  <p className="mt-1 font-mono text-[11px] text-slate-500">
                    hashing · SSIM · localization · contouring
                  </p>
                </div>
              </div>
            )}

            {status === 'error' && (
              <div className="card animate-scale-in flex items-start gap-4 border-rose-500/30">
                <span className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl bg-rose-500/15">
                  <AlertCircle className="text-rose-400" size={19} />
                </span>
                <div>
                  <p className="font-semibold text-rose-300">Comparison failed</p>
                  <p className="mt-1 font-mono text-xs leading-relaxed text-slate-400">{error}</p>
                </div>
              </div>
            )}

            {status === 'done' && result && original && suspected && (
              <ComparativeResultPanel result={result} original={original} suspected={suspected} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
