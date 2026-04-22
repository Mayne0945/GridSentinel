import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // This is crucial: It strips '/api' from the request so FastAPI sees '/status' instead of '/api/status'
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})