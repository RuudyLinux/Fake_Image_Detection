import { useEffect, useState } from 'react';
import { History as HistoryIcon, Trash2, Loader2, ImageOff, Clock } from 'lucide-react';
import Reveal from '../components/Reveal';
import { getHistory, clearHistory } from '../api/detection';
import type { HistoryEntry } from '../types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const VERDICT_STYLE: Record<string, string> = {
  REAL:         'text-emerald-300 bg-emerald-500/10 border-emerald-500/25',
  AUTHENTIC:    'text-emerald-300 bg-emerald-500/10 border-emerald-500/25',
  AI_GENERATED: 'text-rose-300 bg-rose-500/10 border-rose-500/25',
  MANIPULATED:  'text-rose-300 bg-rose-500/10 border-rose-500/25',
  EDITED:       'text-orange-300 bg-orange-500/10 border-orange-500/25',
  UNCERTAIN:    'text-amber-300 bg-amber-500/10 border-amber-500/25',
};

export default function History() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [clearing, setClearing] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      setEntries(await getHistory(100));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleClear = async () => {
    setClearing(true);
    try {
      await clearHistory();
      setEntries([]);
    } finally {
      setClearing(false);
    }
  };

  return (
    <div className="px-6 pb-24 pt-28">
      <div className="mx-auto max-w-6xl">
        <div className="mb-10 flex flex-wrap items-end justify-between gap-4 animate-fade-up">
          <div>
            <span className="eyebrow">
              <HistoryIcon size={12} /> History
            </span>
            <h1 className="mt-4 font-display text-4xl font-bold tracking-tight sm:text-5xl">
              Detection <span className="gradient-text">log</span>
            </h1>
            <p className="mt-3 text-slate-400">
              {entries.length > 0
                ? `${entries.length} past analysis${entries.length === 1 ? '' : 'es'}, newest first.`
                : 'Past blind detections and comparisons appear here.'}
            </p>
          </div>

          {entries.length > 0 && (
            <button
              onClick={handleClear}
              disabled={clearing}
              className="btn-secondary border-rose-500/25 text-sm text-rose-300 hover:border-rose-500/50 hover:bg-rose-500/10"
            >
              {clearing ? <Loader2 size={15} className="animate-spin" /> : <Trash2 size={15} />}
              Clear all
            </button>
          )}
        </div>

        {loading && (
          // Skeletons rather than a spinner — the layout doesn't jump when the
          // real cards land.
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="card space-y-3">
                <div className="h-40 w-full animate-shimmer rounded-xl bg-[linear-gradient(90deg,rgba(255,255,255,.04),rgba(255,255,255,.09),rgba(255,255,255,.04))] bg-[length:200%_100%]" />
                <div className="h-5 w-24 animate-shimmer rounded-md bg-[linear-gradient(90deg,rgba(255,255,255,.04),rgba(255,255,255,.09),rgba(255,255,255,.04))] bg-[length:200%_100%]" />
                <div className="h-3 w-32 animate-shimmer rounded-md bg-[linear-gradient(90deg,rgba(255,255,255,.04),rgba(255,255,255,.09),rgba(255,255,255,.04))] bg-[length:200%_100%]" />
              </div>
            ))}
          </div>
        )}

        {!loading && entries.length === 0 && (
          <div className="card flex min-h-[18rem] flex-col items-center justify-center gap-4 border-dashed text-center">
            <span className="flex h-16 w-16 items-center justify-center rounded-2xl border border-white/10 bg-white/[.03]">
              <ImageOff size={26} className="text-slate-600" />
            </span>
            <div>
              <p className="font-medium text-slate-300">Nothing logged yet</p>
              <p className="mt-1 font-mono text-[11px] text-slate-600">
                Run a detection and it will show up here
              </p>
            </div>
          </div>
        )}

        {!loading && entries.length > 0 && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {entries.map((entry, i) => {
              const style =
                VERDICT_STYLE[entry.prediction ?? ''] ??
                'text-slate-400 bg-white/5 border-white/10';
              return (
                <Reveal key={entry.id} delay={Math.min(i, 8) * 60}>
                  <div className="card card-hover group h-full space-y-3 overflow-hidden">
                    <div className="relative overflow-hidden rounded-xl">
                      {entry.thumbnail_url ? (
                        <img
                          src={`${API_BASE}${entry.thumbnail_url}`}
                          alt="detection thumbnail"
                          className="h-40 w-full object-cover transition-transform duration-700 group-hover:scale-105"
                        />
                      ) : (
                        <div className="flex h-40 w-full items-center justify-center bg-white/[.03]">
                          <ImageOff size={22} className="text-slate-700" />
                        </div>
                      )}
                      <span className="absolute right-2 top-2 rounded-md bg-ink-950/80 px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-slate-400 backdrop-blur">
                        {entry.mode}
                      </span>
                    </div>

                    <div className="flex items-center justify-between gap-2">
                      <span className={`rounded-lg border px-2.5 py-1 font-mono text-[10px] font-semibold uppercase tracking-wider ${style}`}>
                        {entry.prediction?.replace('_', ' ') ?? 'UNKNOWN'}
                      </span>
                      {entry.confidence !== null && (
                        <span className="font-mono text-sm font-semibold tabular-nums text-slate-200">
                          {Math.round(entry.confidence * 100)}%
                        </span>
                      )}
                    </div>

                    <p className="flex items-center gap-1.5 font-mono text-[10px] text-slate-600">
                      <Clock size={10} />
                      {new Date(entry.created_at).toLocaleString()}
                    </p>
                  </div>
                </Reveal>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
