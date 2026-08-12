import { defineConfig } from "vite";

export default defineConfig({
  // onnxruntime-web's wasm/jsep glue .mjs files aren't valid targets for
  // esbuild's dev-time dependency pre-bundling (they self-locate their
  // sibling .wasm binaries at runtime in ways esbuild's transform mishandles)
  // — excluding it avoids the dev server 503-ing on dynamic imports of those
  // files. Standard workaround for onnxruntime-web + Vite.
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
});
