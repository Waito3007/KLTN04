import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
      '@components': fileURLToPath(new URL('./src/components', import.meta.url)),
      '@dashboard': fileURLToPath(new URL('./src/components/Dashboard', import.meta.url)),
      '@taskmanager': fileURLToPath(new URL('./src/components/Dashboard/ProjectTaskManager', import.meta.url)),
    }
  }
})
