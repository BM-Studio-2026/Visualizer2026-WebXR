import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  base: '/Visualizer2026-WebXR/',
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        vr: resolve(__dirname, 'vr.html'),
      },
    },
  },
});
