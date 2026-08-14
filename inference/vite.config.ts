import { resolve } from "node:path";
import { defineConfig } from "vite";

const rootDir = import.meta.dirname;

export default defineConfig({
  // onnxruntime-web's wasm/jsep glue .mjs files aren't valid targets for
  // esbuild's dev-time dependency pre-bundling (they self-locate their
  // sibling .wasm binaries at runtime in ways esbuild's transform mishandles)
  // — excluding it avoids the dev server 503-ing on dynamic imports of those
  // files. Standard workaround for onnxruntime-web + Vite.
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  build: {
    // Three independent pages: index.html (the Spec 4 correctness/profiling
    // harness), viewer.html (the 540p-vs-network-1080p demo on 4 static
    // frames), and live.html (the live renderer + real-time upscaling +
    // real temporal warping demo, importing renderer/ code directly). Vite
    // only builds index.html by default; all three need listing explicitly.
    rollupOptions: {
      input: {
        main: resolve(rootDir, "index.html"),
        viewer: resolve(rootDir, "viewer.html"),
        live: resolve(rootDir, "live.html"),
      },
    },
  },
});
