import { useEffect, useState } from 'react';
import { History as HistoryIcon, Trash2, Loader2, ImageOff } from 'lucide-react';
import { getHistory, clearHistory } from '../api/detection';
import type { HistoryEntry } from '../types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const VERDICT_STYLE: Record<string, string> = {
  REAL: 'text-green-400 bg-green-900/30 border-green-700/50',
  AI_GENERATED: 'text-red-400 bg-red-900/30 border-red-700/50',
  EDITED: 'text-orange-400 bg-orange-900/30 border-orange-700/50',
  AUTHENTIC: 'text-green-400 bg-green-900/30 border-green-700/50',
  MANIPULATED: 'text-orange-400 bg-orange-900/30 border-orange-700/50',
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
    <div className="min-h-screen pt-24 pb-16 px-6">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <div className="flex items-center gap-2 text-purple-400 text-sm font-medium mb-2">
              <HistoryIcon size={14} /> History
            </div>
            <h1 className="text-3xl font-bold text-slate-100">Detection History</h1>
            <p className="text-slate-400 mt-1">Past blind detection and comparison results.</p>
          </div>
          {entries.length > 0 && (
            <button
              onClick={handleClear}
              disabled={clearing}
              className="btn-secondary text-sm text-red-400"
            >
              {clearing ? <Loader2 size={16} className="animate-spin" /> : <Trash2 size={16} />}
              Clear All
            </button>
          )}
        </div>

        {loading && (
          <div className="card h-64 flex items-center justify-center">
            <Loader2 size={32} className="text-purple-500 animate-spin" />
          </div>
        )}

        {!loading && entries.length === 0 && (
          <div className="card h-64 flex flex-col items-center justify-center gap-3 border-dashed">
            <ImageOff size={32} className="text-dark-200" />
            <p className="text-slate-400">No detections yet</p>
          </div>
        )}

        {!loading && entries.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {entries.map((entry) => {
              const style = VERDICT_STYLE[entry.prediction ?? ''] ?? 'text-slate-400 bg-dark-400 border-dark-300';
              return (
                <div key={entry.id} className="card space-y-3">
                  {entry.thumbnail_url ? (
                    <img
                      src={`${API_BASE}${entry.thumbnail_url}`}
                      alt="thumbnail"
                      className="w-full h-40 object-cover rounded-xl"
                    />
                  ) : (
                    <div className="w-full h-40 rounded-xl bg-dark-400 flex items-center justify-center">
                      <ImageOff size={24} className="text-dark-200" />
                    </div>
                  )}
                  <div className="flex items-center justify-between">
                    <span className={`text-xs font-semibold px-2 py-1 rounded-lg border ${style}`}>
                      {entry.prediction?.replace('_', ' ') ?? 'UNKNOWN'}
                    </span>
                    <span className="text-xs text-slate-500 uppercase">{entry.mode}</span>
                  </div>
                  {entry.confidence !== null && (
                    <p className="text-sm text-slate-400">
                      Confidence:{' '}
                      <span className="text-slate-200 font-semibold">
                        {Math.round(entry.confidence * 100)}%
                      </span>
                    </p>
                  )}
                  <p className="text-xs text-slate-600">
                    {new Date(entry.created_at).toLocaleString()}
                  </p>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
