import { useEffect, useRef, useState, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
  /** Stagger within a group, in ms. */
  delay?: number;
  className?: string;
}

/**
 * Fades + lifts its children in when they scroll into view.
 *
 * Unobserves after the first trigger: these are one-shot entrances, and
 * leaving observers attached to every card on a long page is wasted work.
 * Elements already in view on load still animate (the observer fires
 * immediately), which is what makes above-the-fold content feel alive.
 */
export default function Reveal({ children, delay = 0, className = '' }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        setVisible(true);
        observer.unobserve(entry.target);
      },
      { threshold: 0.12, rootMargin: '0px 0px -60px 0px' }
    );

    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className={`reveal ${visible ? 'is-visible' : ''} ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
}
