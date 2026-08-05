import { useEffect, useState } from 'react';
import { ShieldCheck, ShieldAlert, Map, FileDown, Loader2, Crosshair, ArrowLeftRight } from 'lucide-react';
import type { ComparativeResult } from '../types';
import ConfidenceBar from './ConfidenceBar';
import Reveal from './Reveal';
import { useCountUp } from '../hooks/useCountUp';
import { downloadComparativeReport } from '../api/detection';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface Props {
  result: ComparativeResult;
  original: File;
  suspected: File;
}

/** Reveals the suspected image over the original via a draggable wipe.
 *  Far more legible for spotting a local edit than two images side by side. */
function CompareSlider({ originalUrl, suspectedUrl }: { originalUrl: string; suspectedUrl: string }) {
  const [pos, setPos] = useState(50);

  return (
    <div className="relative select-none overflow-hidden rounded-xl border border-white/10">
      <img src={originalUrl} alt="original" className="block max-h-72 w-full object-contain" />

      <div className="absolute inset-0 overflow-hidden" style={{ clipPath: `inset(0 0 0 ${pos}%)` }}>
        <img src={suspectedUrl} alt="suspected" className="block h-full w-full object-contain" />
      </div>

      <div
        className="pointer-events-none absolute inset-y-0 w-0.5 bg-gradient-to-b from-violet-400 via-cyan-400 to-violet-400"
        style={{ left: `${pos}%` }}
      >
        <span className="absolute left-1/2 top-1/2 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-white/25 bg-ink-900/90 backdrop-blur">
          <ArrowLeftRight size={13} className="text-white" />
        </span>
      </div>

      <input
        type="range"
        min={0}
        max={100}
        value={pos}
        onChange={(e) => setPos(Number(e.target.value))}
        aria-label="Compare original and suspected image"
        className="absolute inset-0 h-full w-full cursor-ew-resize opacity-0"
      />

      <span className="pointer-events-none absolute bottom-2 left-2 rounded-md bg-ink-950/80 px-2 py-1 font-mono text-[10px] text-slate-300 backdrop-blur">
        ORIGINAL
      </span>
      <span className="pointer-events-none absolute bottom-2 right-2 rounded-md bg-ink-950/80 px-2 py-1 font-mono text-[10px] text-slate-300 backdrop-blur">
        SUSPECTED
      </span>
    </div>
  );
}

export default function ComparativeResultPanel({ result, original, suspected }: Props) {
  const [downloading, setDownloading] = useState(false);
  const [urls, setUrls] = useState<{ orig: string; susp: string } | null>(null);
  const simPct = useCountUp(Math.round(result.similarity_score * 100), 1300);
  const manipulated = result.is_manipulated;

  useEffect(() => {
    const orig = URL.createObjectURL(original);
    const susp = URL.createObjectURL(suspected);
    setUrls({ orig, susp });
    return () => {
      URL.revokeObjectURL(orig);
      URL.revokeObjectURL(susp);
    };
  }, [original, suspected]);

  const handleDownloadReport = async () => {
    setDownloading(true);
    try {
      const blob = await downloadComparativeReport(original, suspected);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'fakeguard_comparative_report.pdf';
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="space-y-5">
      {/* ---------- verdict ---------- */}
      <div className={`card card-lit noise animate-scale-in overflow-hidden border ${
        manipulated ? 'border-rose-500/30' : 'border-emerald-500/30'
      }`}>
        <div className="flex flex-wrap items-start gap-5">
          <span className={`flex h-16 w-16 flex-shrink-0 items-center justify-center rounded-2xl ${
            manipulated ? 'bg-rose-500/12' : 'bg-emerald-500/12'
          }`}>
            {manipulated
              ? <ShieldAlert size={30} className="text-rose-300" />
              : <ShieldCheck size={30} className="text-emerald-300" />}
          </span>

          <div className="min-w-0 flex-1">
            <p className="label-mono">Comparison result</p>
            <h2 className={`mt-1 font-display text-2xl font-bold tracking-tight sm:text-3xl ${
              manipulated ? 'text-rose-300' : 'text-emerald-300'
            }`}>
              {manipulated ? 'MANIPULATED' : 'AUTHENTIC'}
            </h2>
            <p className="mt-1.5 text-sm text-slate-400">
              Similarity <span className="font-mono font-semibold tabular-nums text-white">{simPct}%</span>
              {result.regions_count > 0 && (
                <> · <span className="font-semibold text-rose-300">
                  {result.regions_count} altered region{result.regions_count > 1 ? 's' : ''}
                </span></>
              )}
            </p>

            {/* Why the verdict landed where it did — including the negative
                case ("diffuse, consistent with re-compression"), which is the
                whole point of the localization-ratio check. */}
            {result.evidence?.reasons?.length ? (
              <ul className="mt-3 space-y-1 rounded-xl border border-white/[.07] bg-white/[.02] p-3">
                {result.evidence.reasons.map((reason) => (
                  <li key={reason} className="font-mono text-[11px] leading-relaxed text-slate-400">
                    → {reason}
                  </li>
                ))}
              </ul>
            ) : null}
          </div>

          <button
            onClick={handleDownloadReport}
            disabled={downloading}
            className="btn-secondary flex-shrink-0 px-4 py-2.5 text-sm"
          >
            {downloading ? <Loader2 size={15} className="animate-spin" /> : <FileDown size={15} />}
            PDF
          </button>
        </div>
      </div>

      {/* ---------- metrics ---------- */}
      <Reveal>
        <div className="card space-y-4">
          <ConfidenceBar
            label="Structural similarity (SSIM)"
            value={result.similarity_score}
            color={result.similarity_score >= 0.9 ? 'green' : result.similarity_score >= 0.7 ? 'yellow' : 'red'}
            delay={80}
          />

          <div className="grid grid-cols-3 gap-3">
            {[
              { k: 'pHash', v: `${result.phash_distance}/64`, hint: 'perceptual' },
              { k: 'dHash', v: `${result.dhash_distance}/64`, hint: 'difference' },
              {
                k: 'Localization',
                v: result.localization_ratio !== undefined ? `${result.localization_ratio.toFixed(1)}×` : '—',
                hint: 'in-region ÷ background',
              },
            ].map(({ k, v, hint }) => (
              <div key={k} className="rounded-xl border border-white/[.06] bg-white/[.02] p-3 text-center">
                <p className="font-mono text-[10px] uppercase tracking-wider text-slate-500">{k}</p>
                <p className="mt-0.5 font-mono text-base font-bold tabular-nums text-white">{v}</p>
                <p className="font-mono text-[9px] text-slate-600">{hint}</p>
              </div>
            ))}
          </div>

          <p className="font-mono text-[10px] leading-relaxed text-slate-600">
            Hash distances are resilient to resize and re-compression (0 = identical).
            Localization ratio separates local edits from diffuse re-compression noise —
            values near 1 mean the difference is spread evenly across the image.
          </p>
        </div>
      </Reveal>

      {/* ---------- wipe comparison ---------- */}
      {urls && (
        <Reveal>
          <div className="card space-y-3">
            <p className="label-mono flex items-center gap-1.5">
              <ArrowLeftRight size={11} className="text-violet-400" /> Drag to compare
            </p>
            <CompareSlider originalUrl={urls.orig} suspectedUrl={urls.susp} />
          </div>
        </Reveal>
      )}

      {/* ---------- diff maps ---------- */}
      {(result.diff_map_url || result.highlighted_url) && (
        <Reveal>
          <div className="grid gap-4 md:grid-cols-2">
            {result.diff_map_url && (
              <div className="card space-y-3">
                <p className="label-mono flex items-center gap-1.5">
                  <Map size={11} className="text-cyan-400" /> Difference map
                </p>
                <img
                  src={`${API_BASE}${result.diff_map_url}`}
                  alt="difference map"
                  className="max-h-56 w-full rounded-xl object-contain"
                />
              </div>
            )}
            {result.highlighted_url && (
              <div className="card space-y-3">
                <p className="label-mono flex items-center gap-1.5">
                  <Crosshair size={11} className="text-rose-400" /> Highlighted regions
                </p>
                <img
                  src={`${API_BASE}${result.highlighted_url}`}
                  alt="highlighted regions"
                  className="max-h-56 w-full rounded-xl object-contain"
                />
              </div>
            )}
          </div>
        </Reveal>
      )}
    </div>
  );
}
