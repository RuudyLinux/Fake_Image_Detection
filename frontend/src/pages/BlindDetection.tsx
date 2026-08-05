import { useState, useEffect } from 'react';
import { ScanSearch, Loader2, AlertCircle, SlidersHorizontal, RotateCcw, Cpu } from 'lucide-react';
import UploadZone from '../components/UploadZone';
import BlindResultPanel from '../components/BlindResultPanel';
import Reveal from '../components/Reveal';
import { detectBlind } from '../api/detection';
import type { BlindDetectionResult, DetectionStatus } from '../types';

/** Purely cosmetic progress narration. The backend returns one response with
 *  no progress stream, so these are timed locally — they describe the real
 *  pipeline stages in order, but they are not live telemetry. */
const STAGES = [
  'Decoding image…',
  'Running model ensemble…',
  'Computing forensic signals…',
  'Fusing evidence…',
  'Estimating uncertainty…',
];

function AnalysingPanel() {
  const [stage, setStage] = useState(0);

  useEffect(() => {
    const id = window.setInterval(
      () => setStage((s) => Math.min(s + 1, STAGES.length - 1)),
      900
    );
    return () => window.clearInterval(id);
  }, []);

  return (
    <div className="card card-lit scanlines flex min-h-[22rem] flex-col items-center justify-center gap-6 overflow-hidden">
      <div className="relative flex h-20 w-20 items-center justify-center">
        <span className="absolute inset-0 rounded-full border border-violet-500/40 animate-pulse-ring" />
        <span className="absolute inset-0 rounded-full border border-cyan-500/30 animate-pulse-ring animation-delay-500" />
        <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500 to-cyan-600">
          <Cpu size={24} className="text-white" />
        </span>
      </div>

      <div className="w-full max-w-xs space-y-2">
        {STAGES.map((s, i) => (
          <div
            key={s}
            className={`flex items-center gap-2.5 font-mono text-xs transition-all duration-500 ${
              i < stage ? 'text-slate-500' : i === stage ? 'text-violet-300' : 'text-slate-700'
            }`}
          >
            <span className={`h-1.5 w-1.5 flex-shrink-0 rounded-full transition-colors duration-500 ${
              i < stage ? 'bg-emerald-500' : i === stage ? 'bg-violet-400 animate-pulse' : 'bg-slate-700'
            }`} />
            {s}
            {i === stage && <span className="animate-blink">▋</span>}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function BlindDetection() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<DetectionStatus>('idle');
  const [result, setResult] = useState<BlindDetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [enableGradcam, setEnableGradcam] = useState(true);
  const [useTta, setUseTta] = useState(true);
  const [allowAbstain, setAllowAbstain] = useState(true);

  const options = { confidenceThreshold, enableGradcam, useTta, allowAbstain };
  const analysing = status === 'analyzing';

  const handleAnalyze = async () => {
    if (!file) return;
    setStatus('analyzing');
    setError(null);
    setResult(null);
    try {
      setResult(await detectBlind(file, options));
      setStatus('done');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Analysis failed. Is the backend running?');
      setStatus('error');
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setStatus('idle');
  };

  const toggles = [
    { checked: enableGradcam, set: setEnableGradcam, label: 'Grad-CAM heatmap', hint: 'Visualise which regions drove the verdict.' },
    { checked: useTta, set: setUseTta, label: 'Test-time augmentation', hint: 'Averages resample-free views. More accurate, slower.' },
    { checked: allowAbstain, set: setAllowAbstain, label: 'Allow “uncertain” verdict', hint: 'Decline to answer rather than guess on ambiguous images.' },
  ];

  return (
    <div className="px-6 pb-24 pt-28">
      <div className="mx-auto max-w-6xl">
        <div className="mb-10 animate-fade-up">
          <span className="eyebrow">
            <ScanSearch size={12} /> Blind detection
          </span>
          <h1 className="mt-4 font-display text-4xl font-bold tracking-tight sm:text-5xl">
            Detect without an <span className="gradient-text">original</span>
          </h1>
          <p className="mt-3 max-w-xl text-slate-400">
            Upload a single image. Three models and six forensic signals analyse it
            independently, then the evidence is fused into one calibrated verdict.
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-5">
          {/* ---------- controls ---------- */}
          <div className="space-y-5 lg:col-span-2">
            <Reveal>
              <div className="card card-lit space-y-4">
                <UploadZone
                  label="Input image"
                  file={file}
                  onFileChange={setFile}
                  scanning={analysing}
                />

                <button
                  onClick={handleAnalyze}
                  disabled={!file || analysing}
                  className="btn-primary w-full"
                >
                  {analysing ? (
                    <><Loader2 size={16} className="animate-spin" /> Analysing…</>
                  ) : (
                    <><ScanSearch size={16} /> Analyse image</>
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
              <div className="card space-y-5">
                <p className="label-mono flex items-center gap-1.5">
                  <SlidersHorizontal size={11} /> Settings
                </p>

                <div>
                  <div className="mb-2 flex items-baseline justify-between">
                    <label htmlFor="confidence-threshold" className="text-sm text-slate-300">
                      Confidence threshold
                    </label>
                    <span className="font-mono text-sm font-semibold tabular-nums text-violet-300">
                      {Math.round(confidenceThreshold * 100)}%
                    </span>
                  </div>
                  <input
                    id="confidence-threshold"
                    type="range"
                    min={0}
                    max={100}
                    value={Math.round(confidenceThreshold * 100)}
                    onChange={(e) => setConfidenceThreshold(Number(e.target.value) / 100)}
                    className="w-full"
                  />
                  <p className="mt-1.5 font-mono text-[10px] leading-relaxed text-slate-600">
                    Verdicts below this are flagged as low-confidence.
                  </p>
                </div>

                <div className="divider" />

                {toggles.map(({ checked, set, label, hint }) => (
                  <label key={label} className="group flex cursor-pointer gap-3">
                    <span className="relative mt-0.5 flex-shrink-0">
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={(e) => set(e.target.checked)}
                        className="peer sr-only"
                      />
                      <span className="block h-5 w-9 rounded-full bg-white/10 transition-colors duration-300 peer-checked:bg-gradient-to-r peer-checked:from-violet-500 peer-checked:to-cyan-500" />
                      <span className="absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform duration-300 peer-checked:translate-x-4" />
                    </span>
                    <span>
                      <span className="block text-sm text-slate-300 transition-colors group-hover:text-white">
                        {label}
                      </span>
                      <span className="mt-0.5 block font-mono text-[10px] leading-relaxed text-slate-600">
                        {hint}
                      </span>
                    </span>
                  </label>
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
                    <ScanSearch size={26} className="text-slate-600" />
                  </span>
                  <div>
                    <p className="font-medium text-slate-300">Awaiting input</p>
                    <p className="mt-1 font-mono text-[11px] text-slate-600">
                      Upload an image to begin analysis
                    </p>
                  </div>
                </div>
              </Reveal>
            )}

            {analysing && <AnalysingPanel />}

            {status === 'error' && (
              <div className="card animate-scale-in flex items-start gap-4 border-rose-500/30">
                <span className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl bg-rose-500/15">
                  <AlertCircle className="text-rose-400" size={19} />
                </span>
                <div>
                  <p className="font-semibold text-rose-300">Analysis failed</p>
                  <p className="mt-1 font-mono text-xs leading-relaxed text-slate-400">{error}</p>
                </div>
              </div>
            )}

            {status === 'done' && result && file && (
              <BlindResultPanel result={result} imageFile={file} reportOptions={options} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
