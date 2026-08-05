import { useEffect, useRef } from 'react';

/**
 * Fixed full-viewport backdrop: drifting aurora blobs (CSS) behind a canvas
 * particle network (JS), with a grid and vignette on top.
 *
 * The particle network is the thematic piece — nodes that link up when they
 * drift close, reading as "signals being correlated", which is what the
 * detector actually does.
 *
 * Performance notes, since this runs on every page for the whole session:
 *  - Particle count scales with viewport area and is hard-capped, so a large
 *    monitor doesn't quadratically increase the O(n^2) link check.
 *  - Canvas is sized to a capped DPR. Uncapped devicePixelRatio on a 3x phone
 *    triples the fill cost for no visible gain at this opacity.
 *  - The whole loop is skipped under prefers-reduced-motion, and cancelled on
 *    unmount / while the tab is hidden.
 */
export default function AnimatedBackground() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReduced) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let frame = 0;
    let width = 0;
    let height = 0;

    type Particle = { x: number; y: number; vx: number; vy: number; r: number };
    let particles: Particle[] = [];

    const LINK_DISTANCE = 130;
    const LINK_DISTANCE_SQ = LINK_DISTANCE * LINK_DISTANCE;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const count = Math.min(Math.round((width * height) / 19000), 90);
      particles = Array.from({ length: count }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.25,
        vy: (Math.random() - 0.5) * 0.25,
        r: Math.random() * 1.6 + 0.6,
      }));
    };

    const draw = () => {
      ctx.clearRect(0, 0, width, height);

      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        // Wrap rather than bounce — bouncing makes particles visibly pile up
        // along the edges over time.
        if (p.x < -20) p.x = width + 20;
        if (p.x > width + 20) p.x = -20;
        if (p.y < -20) p.y = height + 20;
        if (p.y > height + 20) p.y = -20;
      }

      for (let i = 0; i < particles.length; i++) {
        const a = particles[i];
        for (let j = i + 1; j < particles.length; j++) {
          const b = particles[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const distSq = dx * dx + dy * dy;
          if (distSq > LINK_DISTANCE_SQ) continue;
          // Only take the square root for pairs that actually link.
          const alpha = (1 - Math.sqrt(distSq) / LINK_DISTANCE) * 0.22;
          ctx.strokeStyle = `rgba(154, 109, 255, ${alpha})`;
          ctx.lineWidth = 0.6;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }

      for (const p of particles) {
        ctx.fillStyle = 'rgba(186, 156, 255, .5)';
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      }

      frame = requestAnimationFrame(draw);
    };

    const onVisibility = () => {
      cancelAnimationFrame(frame);
      if (!document.hidden) frame = requestAnimationFrame(draw);
    };

    resize();
    frame = requestAnimationFrame(draw);
    window.addEventListener('resize', resize);
    document.addEventListener('visibilitychange', onVisibility);

    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener('resize', resize);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, []);

  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden" aria-hidden="true">
      <div className="absolute inset-0 bg-ink-950" />

      {/* Aurora blobs. Heavy blur + low opacity so they read as ambient light. */}
      <div
        className="absolute -left-40 -top-40 h-[46rem] w-[46rem] rounded-full opacity-[.22] blur-[130px] animate-float"
        style={{ background: 'radial-gradient(circle, #7c3aed, transparent 68%)' }}
      />
      <div
        className="absolute -right-40 top-1/4 h-[38rem] w-[38rem] rounded-full opacity-[.16] blur-[130px] animate-float animation-delay-500"
        style={{ background: 'radial-gradient(circle, #0899b5, transparent 68%)' }}
      />
      <div
        className="absolute bottom-0 left-1/3 h-[34rem] w-[34rem] rounded-full opacity-[.13] blur-[130px] animate-float animation-delay-300"
        style={{ background: 'radial-gradient(circle, #b794ff, transparent 68%)' }}
      />

      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />

      <div className="grid-bg absolute inset-0 opacity-60" />

      {/* Vignette — keeps focus centred and stops the grid reaching the edges. */}
      <div
        className="absolute inset-0"
        style={{
          background:
            'radial-gradient(ellipse at 50% 0%, transparent 20%, rgba(5,5,10,.55) 70%, #05050a 100%)',
        }}
      />
    </div>
  );
}
