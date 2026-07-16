import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

function workspace(path: string): string {
  return fileURLToPath(new URL(`../../../${path}`, import.meta.url));
}

export default defineConfig({
  resolve: {
    alias: [
      {
        find: "@rayvan/local-database/sqlite",
        replacement: workspace("packages/local-database/src/sqlite.ts"),
      },
      {
        find: "@rayvan/daemon-contracts",
        replacement: workspace("packages/daemon-contracts/src/index.ts"),
      },
      {
        find: "@rayvan/daemon-client",
        replacement: workspace("packages/daemon-client/src/index.ts"),
      },
      {
        find: "@rayvan/local-database",
        replacement: workspace("packages/local-database/src/index.ts"),
      },
      {
        find: "@rayvan/findings-engine",
        replacement: workspace("packages/findings-engine/src/index.ts"),
      },
      {
        find: "@rayvan/plugin-sdk",
        replacement: workspace("packages/plugin-sdk/src/index.ts"),
      },
      {
        find: "@rayvan/plugin-example-local",
        replacement: workspace("plugins/example-local/src/index.ts"),
      },
      {
        find: "@rayvan/core",
        replacement: workspace("packages/core/src/index.ts"),
      },
      {
        find: "@rayvan/shared",
        replacement: workspace("packages/shared/src/index.ts"),
      },
    ],
  },
  test: {
    include: ["tests/**/*.test.ts"],
  },
});
