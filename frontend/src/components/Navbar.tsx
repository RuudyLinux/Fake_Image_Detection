import { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ScanLine, Menu, X } from 'lucide-react';

const LINKS = [
  { to: '/', label: 'Home' },
  { to: '/blind', label: 'Blind Detection' },
  { to: '/compare', label: 'Comparative' },
  { to: '/history', label: 'History' },
];

export default function Navbar() {
  const { pathname } = useLocation();
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  // Bare/transparent at the top of the page, solid once content scrolls
  // under it — keeps the hero uncluttered without ever hurting legibility.
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => setMenuOpen(false), [pathname]);

  return (
    <nav
      className={`fixed inset-x-0 top-0 z-50 transition-all duration-500 ${
        scrolled
          ? 'border-b border-white/[.07] bg-ink-950/80 backdrop-blur-xl'
          : 'border-b border-transparent bg-transparent'
      }`}
    >
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
        <Link to="/" className="group flex items-center gap-2.5">
          <span className="relative flex h-9 w-9 items-center justify-center rounded-xl border border-violet-500/30 bg-violet-500/10">
            <ScanLine size={18} className="text-violet-300" />
            <span className="absolute inset-0 rounded-xl border border-violet-400/40 animate-pulse-ring" />
          </span>
          <span className="font-display text-lg font-bold tracking-tight">
            <span className="gradient-text">FakeGuard</span>
            <span className="ml-1 font-mono text-[10px] font-medium uppercase tracking-[.2em] text-slate-500">
              AI
            </span>
          </span>
        </Link>

        {/* Desktop nav. The active pill is a separate absolutely-positioned
            layer so it slides between items instead of hard-cutting. */}
        <div className="relative hidden items-center gap-1 md:flex">
          {LINKS.map(({ to, label }) => {
            const active = pathname === to;
            return (
              <Link
                key={to}
                to={to}
                className={`relative rounded-lg px-4 py-2 text-sm font-medium transition-colors duration-300 ${
                  active ? 'text-white' : 'text-slate-400 hover:text-slate-100'
                }`}
              >
                {active && (
                  <span className="absolute inset-0 -z-10 rounded-lg border border-violet-500/30 bg-gradient-to-r from-violet-600/25 to-cyan-500/15 animate-scale-in" />
                )}
                {label}
              </Link>
            );
          })}
        </div>

        <button
          onClick={() => setMenuOpen((v) => !v)}
          className="btn-ghost md:hidden"
          aria-label={menuOpen ? 'Close menu' : 'Open menu'}
          aria-expanded={menuOpen}
        >
          {menuOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {menuOpen && (
        <div className="border-t border-white/[.07] bg-ink-950/95 px-6 py-3 backdrop-blur-xl md:hidden">
          {LINKS.map(({ to, label }, i) => (
            <Link
              key={to}
              to={to}
              className={`block animate-fade-up rounded-lg px-4 py-3 text-sm font-medium ${
                pathname === to ? 'bg-violet-600/20 text-white' : 'text-slate-400'
              }`}
              style={{ animationDelay: `${i * 50}ms` }}
            >
              {label}
            </Link>
          ))}
        </div>
      )}
    </nav>
  );
}
