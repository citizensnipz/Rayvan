/**
 * Optional host presentation metadata for a plugin.
 * Serializable only — no React nodes, functions, or free-form CSS.
 */

export type PluginThemeSurface = "light" | "dark" | "neutral" | "brand";

export type PluginForegroundMode = "light" | "dark";

/**
 * Controlled theme tokens. Hosts map these to design-system styles.
 * `accentColor` must be a `#RRGGBB` hex string when provided.
 */
export interface PluginThemeDefinition {
  surface: PluginThemeSurface;
  accentColor?: string;
  foregroundMode?: PluginForegroundMode;
}

/**
 * Icon reference for host rendering.
 * Prefer a known `iconId` from the host icon registry; never embed remote URLs.
 */
export interface PluginIconDefinition {
  /** Host-known icon key (e.g. "vercel", "github"). */
  iconId?: string;
  /** Fallback initials when the host has no matching icon. */
  initials?: string;
  /** Accessible label for the icon. */
  label: string;
}

export interface PluginPresentationDefinition {
  icon?: PluginIconDefinition;
  theme?: PluginThemeDefinition;
  /**
   * When true, the host may offer "Add integration" again even if the
   * project already has a connection for this plugin.
   */
  supportsMultipleConnections?: boolean;
}

export const PLUGIN_THEME_SURFACES: readonly PluginThemeSurface[] = [
  "light",
  "dark",
  "neutral",
  "brand",
] as const;

export const PLUGIN_FOREGROUND_MODES: readonly PluginForegroundMode[] = [
  "light",
  "dark",
] as const;

/** `#RRGGBB` only — no alpha, named colors, or CSS functions. */
export const PLUGIN_ACCENT_COLOR_PATTERN = /^#[0-9A-Fa-f]{6}$/;
