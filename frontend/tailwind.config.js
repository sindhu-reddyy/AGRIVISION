/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        forest: {
          900: '#f0fdf4', // Very light green (bg)
          800: '#dcfce7', // Light green
          700: '#15803d', // Dark green text
          100: '#14532d', // Very dark green text
        },
        magic: {
          glow: '#16a34a', // Green glow
          purple: '#22c55e', // Changed purple to bright green for garden feel
        },
        // Add specific text colors if needed
        text: {
          primary: '#052e16',
          secondary: '#166534'
        }
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        serif: ['Cinzel', 'serif'],
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-20px)' },
        }
      }
    },
  },
  plugins: [],
}
