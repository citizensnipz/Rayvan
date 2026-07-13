import { describe, expect, it } from "vitest";
import { manifest as githubManifest } from "../../plugins/github/src/manifest.ts";
import { manifest as vercelManifest } from "../../plugins/vercel/src/manifest.ts";
import { manifest as supabaseManifest } from "../../plugins/supabase/src/manifest.ts";
import { manifest as runpodManifest } from "../../plugins/runpod/src/manifest.ts";

const manifests = [
  githubManifest,
  vercelManifest,
  supabaseManifest,
  runpodManifest,
];

describe("plugin manifests", () => {
  it("exports a manifest from every bundled plugin", () => {
    for (const manifest of manifests) {
      expect(manifest.id).toBeTruthy();
      expect(manifest.name).toBeTruthy();
      expect(manifest.capabilities.length).toBeGreaterThan(0);
    }
  });

  it("uses unique plugin ids", () => {
    const ids = manifests.map((manifest) => manifest.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});
