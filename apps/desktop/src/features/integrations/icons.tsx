import type { CSSProperties } from "react";

import type { IntegrationIconViewModel } from "./view-models.js";
import type { ResolvedIntegrationTheme } from "./theme.js";

interface IntegrationIconProps {
  icon: IntegrationIconViewModel | undefined;
  theme: ResolvedIntegrationTheme;
  size?: number;
}

/**
 * Renders a plugin icon chip. Rayvan has no bundled icon asset set yet, so
 * every plugin renders as an initials chip in its resolved brand color —
 * this also covers the "missing icon" fallback with no special-casing.
 */
export function IntegrationIcon({ icon, theme, size = 40 }: IntegrationIconProps) {
  const label = icon?.label?.trim() || "Integration";
  const initials = (icon?.initials?.trim() || label.slice(0, 2) || "?")
    .slice(0, 3)
    .toUpperCase();

  const style: CSSProperties = {
    width: size,
    height: size,
    borderRadius: "10px",
    display: "grid",
    placeItems: "center",
    background: theme.iconBackground,
    color: theme.iconForeground,
    fontWeight: 700,
    fontSize: size <= 32 ? "0.7rem" : "0.85rem",
    flexShrink: 0,
    userSelect: "none",
  };

  return (
    <div role="img" aria-label={label} style={style}>
      {initials}
    </div>
  );
}
