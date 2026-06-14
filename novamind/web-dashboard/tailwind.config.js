/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0b0f19',
        surface: 'rgba(255, 255, 255, 0.03)',
        border: 'rgba(255, 255, 255, 0.1)',
        accent: '#3b82f6',
        neonPurple: '#8b5cf6',
      }
    },
  },
  plugins: [],
}
