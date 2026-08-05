import { useEffect, useRef, useState } from 'react';

interface Props {
  label: string;
  value: number;
  color?: 'purple' | 'violet' | 'green' | 'red' | 'yellow' | 'blue' | 'cyan';
  /** Stagger when several bars animate as a group, in ms. */
  delay?: number;
  hint?: string;
}

const GRADIENTS: Record<string, string> = {
  purple: 'from-violet-500 to-violet-700',
  violet: 'from-violet-500 to-violet-700',
  green:  'from-emerald-400 to-teal-600',
  red:    'from-rose-500 to-red-700',
  yellow: 'from-amber-400 to-orange-600',
  blue:   'from-sky-400 to-blue-600',
  cyan:   'from-cyan-400 to-cyan-600',
};

export default function ConfidenceBar({ label, value, color = 'violet', delay = 0, hint }: Props) {
  const pct = Math.round(value * 100);
  const [width, setWidth] = useState(0);
  const ref = useRef<HTMLDivElement | null>(null);

  // Fill from 0 once the bar scrolls into view, rather than on mount — a bar
  // that finished animating while off-screen just appears already full.
  useEffect(() => {
    const node = ref.current;
    if (!node) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        const timer = window.setTimeout(() => setWidth(pct), delay);
        observer.unobserve(entry.target);
        return () => window.clearTimeout(timer);
      },
      { threshold: 0.3 }
    );

    observer.observe(node);
    return () => observer.disconnect();
  }, [pct, delay]);

  return (
    <div ref={ref} className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-sm text-slate-400">{label}</span>
        <span className="font-mono text-sm font-semibold tabular-nums text-slate-100">{pct}%</span>
      </div>

      <div className="relative h-2 overflow-hidden rounded-full bg-white/[.06]">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${GRADIENTS[color] ?? GRADIENTS.violet} transition-[width] duration-1000 ease-out`}
          style={{ width: `${width}%` }}
        >
          {/* Travelling highlight — only worth rendering on bars wide enough to see it. */}
          {width > 8 && (
            <div className="h-full w-full animate-shimmer bg-[linear-gradient(90deg,transparent,rgba(255,255,255,.45),transparent)] bg-[length:200%_100%]" />
          )}
        </div>
      </div>

      {hint && <p className="font-mono text-[10px] leading-relaxed text-slate-600">{hint}</p>}
    </div>
  );
}
