import { useRef, useState, useCallback, useEffect } from 'react';
import { UploadCloud, X, RefreshCw, ImageIcon } from 'lucide-react';

interface Props {
  label: string;
  file: File | null;
  onFileChange: (f: File | null) => void;
  accept?: string;
  /** Runs the scan-line overlay — set while the backend is working. */
  scanning?: boolean;
  accent?: 'violet' | 'cyan';
}

const ACCENTS = {
  violet: { text: 'text-violet-300', border: 'border-violet-500/60', bg: 'bg-violet-500/10', line: 'via-violet-400' },
  cyan:   { text: 'text-cyan-300',   border: 'border-cyan-500/60',   bg: 'bg-cyan-500/10',   line: 'via-cyan-400' },
};

export default function UploadZone({
  label, file, onFileChange, accept = 'image/*', scanning = false, accent = 'violet',
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const a = ACCENTS[accent];

  // Object URLs must be revoked, or every re-selected file leaks its blob for
  // the lifetime of the page. The previous version created one inline during
  // render and never released it.
  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const dropped = e.dataTransfer.files[0];
      if (dropped && dropped.type.startsWith('image/')) onFileChange(dropped);
    },
    [onFileChange]
  );

  return (
    <div className="space-y-2.5">
      <p className="label-mono">{label}</p>

      {file && preview ? (
        <div className="group relative overflow-hidden rounded-xl border border-white/10">
          <img src={preview} alt="preview" className="h-56 w-full object-cover" />

          {scanning && (
            <>
              <div className="pointer-events-none absolute inset-0 overflow-hidden">
                <div className={`h-24 w-full bg-gradient-to-b from-transparent ${a.line} to-transparent opacity-50 blur-sm animate-scan`} />
              </div>
              <div className="scanlines pointer-events-none absolute inset-0" />
            </>
          )}

          <div className="absolute inset-0 flex items-center justify-center gap-3 bg-ink-950/70 opacity-0 backdrop-blur-sm transition-opacity duration-300 group-hover:opacity-100">
            <button
              onClick={() => inputRef.current?.click()}
              className="rounded-xl border border-white/15 bg-white/10 p-2.5 text-white transition-all hover:scale-110 hover:bg-white/20"
              title="Replace image"
            >
              <RefreshCw size={17} />
            </button>
            <button
              onClick={() => onFileChange(null)}
              className="rounded-xl border border-rose-500/30 bg-rose-500/20 p-2.5 text-rose-300 transition-all hover:scale-110 hover:bg-rose-500/30"
              title="Remove image"
            >
              <X size={17} />
            </button>
          </div>

          <div className="absolute inset-x-2 bottom-2 flex items-center gap-2 rounded-lg border border-white/10 bg-ink-950/85 px-3 py-1.5 backdrop-blur-md">
            <ImageIcon size={12} className={a.text} />
            <span className="truncate font-mono text-[11px] text-slate-300">{file.name}</span>
            <span className="ml-auto flex-shrink-0 font-mono text-[10px] text-slate-500">
              {(file.size / 1024).toFixed(0)} KB
            </span>
          </div>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          className={`relative flex h-56 w-full flex-col items-center justify-center gap-3 overflow-hidden rounded-xl border-2 border-dashed transition-all duration-300 ${
            dragging
              ? `${a.border} ${a.bg} scale-[1.015]`
              : 'border-white/10 hover:border-white/25 hover:bg-white/[.03]'
          }`}
        >
          <div className="grid-bg absolute inset-0 opacity-40" />

          <span className={`relative flex h-14 w-14 items-center justify-center rounded-2xl border transition-all duration-300 ${
            dragging ? `${a.border} ${a.bg} scale-110` : 'border-white/10 bg-white/5'
          }`}>
            <UploadCloud size={22} className={dragging ? a.text : 'text-slate-400'} />
            {dragging && <span className={`absolute inset-0 rounded-2xl border ${a.border} animate-pulse-ring`} />}
          </span>

          <div className="relative text-center">
            <p className="text-sm font-semibold text-slate-200">
              {dragging ? 'Release to upload' : 'Drop an image here'}
            </p>
            <p className="mt-0.5 font-mono text-[11px] text-slate-500">
              or click to browse · PNG JPG WEBP · max 20 MB
            </p>
          </div>
        </button>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => onFileChange(e.target.files?.[0] ?? null)}
      />
    </div>
  );
}
