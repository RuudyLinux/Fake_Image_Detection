import { useCallback, useRef, type MouseEvent } from 'react';

interface TiltOptions {
  /** Peak rotation at the card's corners, in degrees. */
  max?: number;
  scale?: number;
}

/**
 * Pointer-following 3D tilt for a card.
 *
 * Writes transforms directly to the node rather than through React state —
 * mousemove fires far faster than a useful render cadence, and re-rendering
 * a whole card subtree per pointer sample is what makes naive tilt
 * implementations feel sticky.
 *
 * Also publishes the pointer position as CSS custom properties (--mx/--my) so
 * a child element can render a spotlight that tracks the cursor without any
 * extra listeners.
 */
export function useTilt({ max = 8, scale = 1.02 }: TiltOptions = {}) {
  const ref = useRef<HTMLDivElement | null>(null);

  const onMouseMove = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      const node = ref.current;
      if (!node) return;
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

      const rect = node.getBoundingClientRect();
      const px = (e.clientX - rect.left) / rect.width;
      const py = (e.clientY - rect.top) / rect.height;

      const rotateY = (px - 0.5) * 2 * max;
      const rotateX = -(py - 0.5) * 2 * max;

      node.style.transform =
        `perspective(900px) rotateX(${rotateX.toFixed(2)}deg) rotateY(${rotateY.toFixed(2)}deg) scale(${scale})`;
      node.style.setProperty('--mx', `${px * 100}%`);
      node.style.setProperty('--my', `${py * 100}%`);
    },
    [max, scale]
  );

  const onMouseLeave = useCallback(() => {
    const node = ref.current;
    if (!node) return;
    node.style.transform = '';
  }, []);

  return { ref, onMouseMove, onMouseLeave };
}
