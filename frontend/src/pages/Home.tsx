import { useNavigate } from 'react-router-dom';
import type { LucideIcon } from 'lucide-react';
import {
  ScanSearch, GitCompare, Sparkles, Layers, Waves, Fingerprint,
  ShieldQuestion, Gauge, ArrowRight, Activity,
} from 'lucide-react';
import Reveal from '../components/Reveal';
import { useCountUp } from '../hooks/useCountUp';
import { useTilt } from '../hooks/useTilt';

/** Metrics shown on the landing page.
 *  These are the project's real measured numbers, not marketing figures —
 *  see PROJECT_SPEC.md's Honest Status and checkpoints/*.metrics.json. Keep
 *  them in sync when a checkpoint is promoted; a stale number here is a
 *  claim the system can no longer back up. */
const STATS: { value: number; decimals: number; suffix: string; label: string; note: string }[] = [
  { value: 97.3, decimals: 1, suffix: '%', label: 'Ensemble accuracy', note: 'pristine 450-image sample' },
  { value: 3,    decimals: 0, suffix: '',  label: 'Models ensembled',  note: '2 fine-tuned CNNs + frozen CLIP' },
  { value: 0.978, decimals: 3, suffix: '', label: 'Comparative F1',    note: '3396 CASIA ground-truth pairs' },
  { value: 6,    decimals: 0, suffix: '',  label: 'Forensic signals',  note: 'FFT · ELA · noise · texture · edge · GAN' },
];

const PIPELINE: { icon: LucideIcon; title: string; body: string; accent: string }[] = [
  { icon: Layers, title: '3-model ensemble', accent: 'from-violet-500 to-violet-700',
    body: 'EfficientNet-B4 and ConvNeXt-Tiny at different resolutions, plus a frozen CLIP probe that generalizes to generators the CNNs never saw.' },
  { icon: Waves, title: 'Forensic heuristics', accent: 'from-cyan-500 to-cyan-700',
    body: 'Spectral decay, GAN upsampling periodicity, noise residual, texture uniformity, edge statistics and error-level analysis — computed independently of any model.' },
  { icon: ShieldQuestion, title: 'Principled abstention', accent: 'from-amber-500 to-orange-600',
    body: 'Predictive entropy is split into genuine ambiguity and model disagreement. When the models disagree, the verdict is UNCERTAIN — not a guess.' },
  { icon: Gauge, title: 'Calibrated confidence', accent: 'from-emerald-500 to-teal-600',
    body: 'Temperature scaling fitted on held-out data, so a stated 90% actually behaves like 90% instead of raw softmax overconfidence.' },
];

function StatCard({ stat, index }: { stat: (typeof STATS)[number]; index: number }) {
  const value = useCountUp(stat.value, 1600, stat.decimals);
  return (
    <Reveal delay={index * 90}>
      <div className="card card-hover scanlines h-full overflow-hidden text-center">
        <p className="font-display text-4xl font-bold">
          <span className="gradient-text">
            {value.toFixed(stat.decimals)}
            {stat.suffix}
          </span>
        </p>
        <p className="mt-2 text-sm font-semibold text-slate-200">{stat.label}</p>
        <p className="mt-1 font-mono text-[10px] leading-relaxed text-slate-500">{stat.note}</p>
      </div>
    </Reveal>
  );
}

function ModeCard({
  icon: Icon, title, body, features, accent, glow, onClick, cta,
}: {
  icon: LucideIcon; title: string; body: string; features: string[];
  accent: string; glow: string; onClick: () => void; cta: string;
}) {
  const tilt = useTilt({ max: 6 });

  return (
    <div
      ref={tilt.ref}
      onMouseMove={tilt.onMouseMove}
      onMouseLeave={tilt.onMouseLeave}
      className="group relative h-full transition-transform duration-300 ease-out will-change-transform"
    >
      <button
        onClick={onClick}
        className="card card-lit noise relative h-full w-full overflow-hidden text-left"
      >
        {/* Cursor-tracking spotlight, positioned from the --mx/--my vars the
            tilt hook writes. */}
        <span
          className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-500 group-hover:opacity-100"
          style={{
            background: `radial-gradient(420px circle at var(--mx, 50%) var(--my, 50%), ${glow}, transparent 70%)`,
          }}
        />

        <div className="relative space-y-5">
          <div className="flex items-start justify-between">
            <span className={`flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br ${accent} shadow-lg`}>
              <Icon size={24} className="text-white" />
            </span>
            <ArrowRight
              size={20}
              className="translate-x-0 text-slate-600 transition-all duration-300 group-hover:translate-x-1 group-hover:text-white"
            />
          </div>

          <div>
            <h3 className="font-display text-xl font-bold text-white">{title}</h3>
            <p className="mt-1.5 text-sm leading-relaxed text-slate-400">{body}</p>
          </div>

          <ul className="space-y-2">
            {features.map((f, i) => (
              <li
                key={f}
                className="flex items-center gap-2.5 text-sm text-slate-400 transition-transform duration-300 group-hover:translate-x-1"
                style={{ transitionDelay: `${i * 40}ms` }}
              >
                <span className={`h-1.5 w-1.5 flex-shrink-0 rounded-full bg-gradient-to-br ${accent}`} />
                {f}
              </li>
            ))}
          </ul>

          <div className="flex items-center gap-2 pt-1 font-mono text-xs uppercase tracking-[.16em] text-slate-300">
            {cta}
            <span className="h-px flex-1 bg-gradient-to-r from-white/25 to-transparent" />
          </div>
        </div>
      </button>
    </div>
  );
}

export default function Home() {
  const navigate = useNavigate();

  return (
    <div className="relative px-6 pb-24 pt-32">
      <div className="mx-auto max-w-6xl">
        {/* ---------- hero ---------- */}
        <div className="mx-auto max-w-3xl text-center">
          <div className="animate-fade-up">
            <span className="eyebrow">
              <Sparkles size={12} />
              AI Image Forensics
            </span>
          </div>

          <h1 className="mt-6 animate-fade-up animation-delay-100 font-display text-5xl font-bold leading-[1.08] tracking-tight text-balance sm:text-6xl md:text-7xl">
            Find the{' '}
            <span className="glitch gradient-text" data-text="forgery">
              forgery
            </span>
            <br />
            before it finds you
          </h1>

          <p className="mx-auto mt-6 max-w-xl animate-fade-up animation-delay-200 text-lg leading-relaxed text-slate-400 text-balance">
            A calibrated model ensemble and six independent forensic signals analyse
            every pixel — and say <span className="font-semibold text-slate-200">&ldquo;uncertain&rdquo;</span>{' '}
            when the evidence doesn&rsquo;t support a verdict.
          </p>

          <div className="mt-9 flex animate-fade-up animation-delay-300 flex-col items-center justify-center gap-3 sm:flex-row">
            <button onClick={() => navigate('/blind')} className="btn-primary w-full sm:w-auto">
              <ScanSearch size={17} />
              Analyse an image
            </button>
            <button onClick={() => navigate('/compare')} className="btn-secondary w-full sm:w-auto">
              <GitCompare size={17} />
              Compare two images
            </button>
          </div>

          <div className="mt-7 flex animate-fade-in animation-delay-500 items-center justify-center gap-2 font-mono text-[11px] text-slate-600">
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500" />
            </span>
            REAL · AI_GENERATED · EDITED · UNCERTAIN
          </div>
        </div>

        {/* ---------- mode cards ---------- */}
        <div className="mt-24 grid gap-6 md:grid-cols-2">
          <Reveal>
            <ModeCard
              icon={ScanSearch}
              title="Detect without original"
              body="One image in. The ensemble and every forensic signal run against it, with a Grad-CAM heatmap showing what drove the call."
              features={['3-model calibrated ensemble', 'FFT & GAN fingerprint analysis', 'Grad-CAM explainability', 'Uncertainty breakdown']}
              accent="from-violet-500 to-violet-700"
              glow="rgba(154,109,255,.16)"
              cta="Start blind detection"
              onClick={() => navigate('/blind')}
            />
          </Reveal>
          <Reveal delay={120}>
            <ModeCard
              icon={GitCompare}
              title="Compare with original"
              body="Two images in. Localization-ratio analysis separates real edits from harmless re-compression, then pinpoints the altered regions."
              features={['Perceptual hash distance', 'Localization-ratio scoring', 'Altered-region highlighting', 'Difference heatmap']}
              accent="from-cyan-500 to-cyan-700"
              glow="rgba(56,220,245,.14)"
              cta="Start comparison"
              onClick={() => navigate('/compare')}
            />
          </Reveal>
        </div>

        {/* ---------- stats ---------- */}
        <div className="mt-20">
          <Reveal>
            <div className="mb-8 flex items-center gap-3">
              <Activity size={15} className="text-violet-400" />
              <h2 className="label-mono">Measured, not claimed</h2>
              <span className="h-px flex-1 bg-gradient-to-r from-white/10 to-transparent" />
            </div>
          </Reveal>
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            {STATS.map((stat, i) => (
              <StatCard key={stat.label} stat={stat} index={i} />
            ))}
          </div>
        </div>

        {/* ---------- pipeline ---------- */}
        <div className="mt-24">
          <Reveal>
            <div className="mb-10 text-center">
              <span className="eyebrow">
                <Fingerprint size={12} /> How it works
              </span>
              <h2 className="mt-5 font-display text-3xl font-bold tracking-tight sm:text-4xl">
                Four layers of <span className="gradient-text">evidence</span>
              </h2>
              <p className="mx-auto mt-3 max-w-xl text-slate-400 text-balance">
                No single signal decides anything. Each layer can be inspected, and
                each one can disagree with the others.
              </p>
            </div>
          </Reveal>

          <div className="grid gap-5 sm:grid-cols-2">
            {PIPELINE.map(({ icon: Icon, title, body, accent }, i) => (
              <Reveal key={title} delay={i * 90}>
                <div className="card card-hover group h-full">
                  <div className="flex gap-4">
                    <span className={`flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-xl bg-gradient-to-br ${accent} transition-transform duration-500 group-hover:scale-110 group-hover:rotate-6`}>
                      <Icon size={19} className="text-white" />
                    </span>
                    <div>
                      <div className="flex items-baseline gap-2">
                        <span className="font-mono text-[10px] text-slate-600">
                          0{i + 1}
                        </span>
                        <h3 className="font-display font-bold text-white">{title}</h3>
                      </div>
                      <p className="mt-1.5 text-sm leading-relaxed text-slate-400">{body}</p>
                    </div>
                  </div>
                </div>
              </Reveal>
            ))}
          </div>
        </div>

        {/* ---------- honesty note ----------
            Deliberately on the landing page rather than buried. Every
            competing tool claims perfect accuracy; the credible move is to
            state the limit up front. */}
        <Reveal>
          <div className="card card-lit noise mt-20 overflow-hidden text-center">
            <span className="eyebrow">
              <ShieldQuestion size={12} /> An honest note
            </span>
            <h3 className="mt-5 font-display text-2xl font-bold sm:text-3xl">
              No detector is 100% accurate. Including this one.
            </h3>
            <p className="mx-auto mt-4 max-w-2xl leading-relaxed text-slate-400 text-balance">
              Generative models are trained to match the distribution of real photographs,
              so the two overlap — and where they overlap, perfect separation is not
              achievable by any system. This one is built to know when it is out of its
              depth and say so, rather than return a confident answer it cannot support.
            </p>
            <div className="mt-6 flex flex-wrap items-center justify-center gap-x-6 gap-y-2 font-mono text-[11px] text-slate-500">
              <span>abstains on low confidence</span>
              <span className="text-slate-700">·</span>
              <span>abstains on model disagreement</span>
              <span className="text-slate-700">·</span>
              <span>reports calibrated probability</span>
            </div>
          </div>
        </Reveal>
      </div>
    </div>
  );
}
