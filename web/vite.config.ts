import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Built into the package so FastAPI can serve it as static files — one
  // process in production, no separate node server to keep alive.
  build: { outDir: '../cef/api/webdist', emptyOutDir: true },
  server: {
    port: 5199,
    proxy: {
      // The SSE endpoint must not be buffered by the dev proxy, or the stage
      // events all arrive at once and the whole live view is pointless.
      '/api': { target: 'http://127.0.0.1:8124', changeOrigin: true, ws: false },
    },
  },
})
