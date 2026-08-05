import { useEffect, useRef, useState } from 'react';

/**
 * Animates a number from 0 to `target` on mount, and again whenever `target`
 * changes. Eased with an ease-out cubic so it decelerates into the final
 * value instead of stopping dead.
 *
 * Driven by requestAnimationFrame against wall-clock time rather than a
 * per-frame increment, so the duration is honest on any refresh rate — a
 * fixed step-per-frame runs twice as fast on a 120Hz display.
 */
export function useCountUp(target: number, duration = 1200, decimals = 0): number {
  const [value, setValue] = useState(0);
  const frameRef = useRef(0);

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setValue(target);
      return;
    }

    const start = performance.now();
    const factor = 10 ** decimals;

    const tick = (now: number) => {
      const t = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(Math.round(target * eased * factor) / factor);
      if (t < 1) frameRef.current = requestAnimationFrame(tick);
    };

    frameRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameRef.current);
  }, [target, duration, decimals]);

  return value;
}
