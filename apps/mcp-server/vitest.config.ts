import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      "@rayvan/daemon-client": fileURLToPath(
        new URL("../../packages/daemon-client/src/index.ts", import.meta.url),
      ),
      "@rayvan/daemon-contracts": fileURLToPath(
        new URL("../../packages/daemon-contracts/src/index.ts", import.meta.url),
      ),
    },
  },
  test: {
    include: ["tests/**/*.test.ts"],
  },
});
