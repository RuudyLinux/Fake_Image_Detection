/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Forensic heatmap palette: violet (primary) -> cyan (analysis) ->
        // amber/rose (findings). Chosen so verdict colours read as a spectrum
        // rather than arbitrary UI accents.
        violet: {
          300: '#d0bcff', 400: '#b794ff', 500: '#9a6dff',
          600: '#7c3aed', 700: '#6425d0', 800: '#4c1d95', 900: '#2e1065',
        },
        cyan: {
          300: '#7ff0ff', 400: '#38dcf5', 500: '#0ec3e0', 600: '#0899b5',
        },
        ink: {
          950: '#050509', 900: '#08080f', 850: '#0b0b16',
          800: '#0f0f1c', 700: '#151527', 600: '#1c1c33',
          500: '#252542', 400: '#323256', 300: '#43436e',
        },
        // Kept so any un-migrated markup still resolves to sane values.
        dark: {
          900: '#05050a', 800: '#08080f', 700: '#0f0f1c', 600: '#151527',
          500: '#1c1c33', 400: '#252542', 300: '#323256', 200: '#43436e',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['"Space Grotesk"', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(24px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'scale-in': {
          '0%': { opacity: '0', transform: 'scale(.94)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        // Vertical sweep used on upload zones and analysing states — reads as
        // "the image is being scanned", which is literally what's happening.
        scan: {
          '0%': { transform: 'translateY(-100%)', opacity: '0' },
          '10%,90%': { opacity: '1' },
          '100%': { transform: 'translateY(1000%)', opacity: '0' },
        },
        'scan-x': {
          '0%': { transform: 'translateX(-120%)' },
          '100%': { transform: 'translateX(120%)' },
        },
        float: {
          '0%,100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-12px)' },
        },
        'pulse-ring': {
          '0%': { transform: 'scale(.85)', opacity: '.7' },
          '100%': { transform: 'scale(1.6)', opacity: '0' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        'gradient-pan': {
          '0%,100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
        marquee: {
          '0%': { transform: 'translateX(0)' },
          '100%': { transform: 'translateX(-50%)' },
        },
        blink: {
          '0%,49%': { opacity: '1' },
          '50%,100%': { opacity: '0' },
        },
      },
      animation: {
        'fade-up': 'fade-up .7s cubic-bezier(.22,1,.36,1) both',
        'fade-in': 'fade-in .6s ease both',
        'scale-in': 'scale-in .5s cubic-bezier(.22,1,.36,1) both',
        scan: 'scan 2.6s cubic-bezier(.4,0,.2,1) infinite',
        'scan-x': 'scan-x 2s ease-in-out infinite',
        float: 'float 6s ease-in-out infinite',
        'pulse-ring': 'pulse-ring 2s cubic-bezier(.4,0,.6,1) infinite',
        shimmer: 'shimmer 2.4s linear infinite',
        'gradient-pan': 'gradient-pan 8s ease infinite',
        marquee: 'marquee 32s linear infinite',
        blink: 'blink 1.1s step-end infinite',
      },
    },
  },
  plugins: [],
}
