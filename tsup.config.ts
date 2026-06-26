import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts", "src/proxy.ts", "src/cli.ts"],
  format: ["esm", "cjs"],
  dts: { entry: ["src/index.ts", "src/proxy.ts"] },
  sourcemap: true,
  clean: true,
  treeshake: true,
  target: "es2021",
  shims: true,
  outExtension({ format }) {
    return { js: format === "cjs" ? ".cjs" : ".js" };
  },
});
