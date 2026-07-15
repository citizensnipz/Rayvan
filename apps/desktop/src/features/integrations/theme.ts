import { PLUGIN_ACCENT_COLOR_PATTERN } from "@rayvan/plugin-sdk";

import type { IntegrationThemeViewModel } from "./view-models.js";

/**
 * Resolves a plugin's brand `presentation.theme` into concrete CSS values
 * for the host UI. This is the single place that maps
 * `PluginThemeSurface` -> colors — cards/detail views must never branch on
 * provider id.
 *
 * Accent colors are re-validated here so a corrupted `manifestSnapshot`
 * cannot inject arbitrary CSS values.
 */
export interface ResolvedIntegrationTheme {
  iconBackground: string;
  iconForeground: string;
  accentColor: string;
}

const DEFAULT_FOREGROUND_LIGHT = "#f8fafc";
const DEFAULT_FOREGROUND_DARK = "#0f172a";
const DARK_SURFACE_BG = "#111827";
const LIGHT_SURFACE_BG = "#f1f5f9";
const NEUTRAL_SURFACE_BG = "var(--color-surface-muted)";
const FALLBACK_ACCENT = "var(--color-border-strong)";

export function sanitizeAccentColor(
  accentColor: string | undefined,
): string | undefined {
  if (typeof accentColor !== "string") {
    return undefined;
  }
  const trimmed = accentColor.trim();
  return PLUGIN_ACCENT_COLOR_PATTERN.test(trimmed) ? trimmed : undefined;
}

function contrastForeground(hex: string): string {
  const value = hex.slice(1);
  const r = Number.parseInt(value.slice(0, 2), 16);
  const g = Number.parseInt(value.slice(2, 4), 16);
  const b = Number.parseInt(value.slice(4, 6), 16);
  // Relative luminance threshold — prefer dark text on light chips.
  const luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  return luminance > 0.6 ? DEFAULT_FOREGROUND_DARK : DEFAULT_FOREGROUND_LIGHT;
}

export function resolveIntegrationTheme(
  theme: IntegrationThemeViewModel | undefined,
): ResolvedIntegrationTheme {
  const surface = theme?.surface ?? "neutral";
  const accent = sanitizeAccentColor(theme?.accentColor);
  const foregroundMode = theme?.foregroundMode ?? "dark";
  const modeForeground =
    foregroundMode === "light" ? DEFAULT_FOREGROUND_LIGHT : DEFAULT_FOREGROUND_DARK;

  switch (surface) {
    case "brand":
      return {
        iconBackground: accent ?? "var(--color-nav-active)",
        iconForeground: accent ? contrastForeground(accent) : modeForeground,
        accentColor: accent ?? FALLBACK_ACCENT,
      };
    case "dark":
      // Surface drives the chip; accent is brand highlight only.
      return {
        iconBackground: DARK_SURFACE_BG,
        iconForeground: DEFAULT_FOREGROUND_LIGHT,
        accentColor: accent ?? DARK_SURFACE_BG,
      };
    case "light":
      return {
        iconBackground: LIGHT_SURFACE_BG,
        iconForeground: DEFAULT_FOREGROUND_DARK,
        accentColor: accent ?? FALLBACK_ACCENT,
      };
    case "neutral":
    default:
      return {
        iconBackground: accent ?? NEUTRAL_SURFACE_BG,
        iconForeground: accent ? contrastForeground(accent) : modeForeground,
        accentColor: accent ?? FALLBACK_ACCENT,
      };
  }
}
