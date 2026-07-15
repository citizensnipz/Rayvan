import { describe, expect, it } from "vitest";

import { resolveIntegrationTheme, sanitizeAccentColor } from "./theme.js";

describe("resolveIntegrationTheme", () => {
  it("does not throw and returns sensible defaults when no theme is provided", () => {
    expect(() => resolveIntegrationTheme(undefined)).not.toThrow();
    const resolved = resolveIntegrationTheme(undefined);
    expect(resolved.iconBackground).toBeTruthy();
    expect(resolved.iconForeground).toBeTruthy();
    expect(resolved.accentColor).toBeTruthy();
  });

  it("resolves brand surfaces using a sanitized accent color when present", () => {
    const resolved = resolveIntegrationTheme({
      surface: "brand",
      accentColor: "#362D59",
      foregroundMode: "light",
    });
    expect(resolved.iconBackground).toBe("#362D59");
    expect(resolved.accentColor).toBe("#362D59");
  });

  it("ignores invalid accent CSS values", () => {
    expect(sanitizeAccentColor("url(javascript:alert(1))")).toBeUndefined();
    const resolved = resolveIntegrationTheme({
      surface: "brand",
      accentColor: "red",
    });
    expect(resolved.iconBackground).not.toBe("red");
  });

  it("keeps dark-surface chips readable when accent is near-white", () => {
    const resolved = resolveIntegrationTheme({
      surface: "dark",
      accentColor: "#FFFFFF",
      foregroundMode: "light",
    });
    expect(resolved.iconBackground).toBe("#111827");
    expect(resolved.iconForeground).toBe("#f8fafc");
    expect(resolved.accentColor).toBe("#FFFFFF");
  });

  it("resolves every declared theme surface without throwing", () => {
    const surfaces: Array<"light" | "dark" | "neutral" | "brand"> = [
      "light",
      "dark",
      "neutral",
      "brand",
    ];
    for (const surface of surfaces) {
      expect(() => resolveIntegrationTheme({ surface })).not.toThrow();
    }
  });
});
