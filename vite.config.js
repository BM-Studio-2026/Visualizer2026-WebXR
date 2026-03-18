import { defineConfig } from 'vite';

export default defineConfig({
  base: '/Visualizer2026-WebXR/',
  build: {
    rollupOptions: {
      input: {
        main: 'index.html',
        vr: 'vr.html'
      }
    }
  }
});