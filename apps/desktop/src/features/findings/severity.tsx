import type { CSSProperties, ReactElement, ReactNode, SVGProps } from "react";
import type { FindingSeverity } from "@rayvan/core";

import { SEVERITY_LABELS } from "./view-models.js";

type IconProps = SVGProps<SVGSVGElement>;

export const SEVERITY_COLORS: Record<FindingSeverity, string> = {
  critical: "#9f1239",
  error: "#dc2626",
  warning: "#d97706",
  info: "#2563eb",
};

function SeveritySvg({ children, ...props }: IconProps & { children: ReactNode }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {children}
    </svg>
  );
}

function CriticalIcon(props: IconProps) {
  return (
    <SeveritySvg {...props}>
      <>
        <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2" />
        <line x1="12" x2="12" y1="8" y2="12" />
        <line x1="12" x2="12.01" y1="16" y2="16" />
      </>
    </SeveritySvg>
  );
}

function ErrorIcon(props: IconProps) {
  return (
    <SeveritySvg {...props}>
      <>
        <circle cx="12" cy="12" r="10" />
        <line x1="15" x2="9" y1="9" y2="15" />
        <line x1="9" x2="15" y1="9" y2="15" />
      </>
    </SeveritySvg>
  );
}

function WarningIcon(props: IconProps) {
  return (
    <SeveritySvg {...props}>
      <>
        <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
        <line x1="12" x2="12" y1="9" y2="13" />
        <line x1="12" x2="12.01" y1="17" y2="17" />
      </>
    </SeveritySvg>
  );
}

function InfoIcon(props: IconProps) {
  return (
    <SeveritySvg {...props}>
      <>
        <circle cx="12" cy="12" r="10" />
        <line x1="12" x2="12" y1="16" y2="12" />
        <line x1="12" x2="12.01" y1="8" y2="8" />
      </>
    </SeveritySvg>
  );
}

const SEVERITY_ICONS: Record<FindingSeverity, (props: IconProps) => ReactElement> = {
  critical: CriticalIcon,
  error: ErrorIcon,
  warning: WarningIcon,
  info: InfoIcon,
};

export function SeverityIcon({
  severity,
  ...props
}: IconProps & { severity: FindingSeverity }) {
  const Component = SEVERITY_ICONS[severity];
  return <Component {...props} />;
}

const badgeStyle: CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: "0.35rem",
  fontWeight: 600,
  lineHeight: 1.2,
};

interface SeverityBadgeProps {
  severity: FindingSeverity;
  label?: string;
  size?: "sm" | "md";
}

export function SeverityBadge({
  severity,
  label = SEVERITY_LABELS[severity],
  size = "sm",
}: SeverityBadgeProps) {
  const color = SEVERITY_COLORS[severity];
  return (
    <span
      aria-label={`Severity: ${label}`}
      style={{
        ...badgeStyle,
        color,
        fontSize: size === "md" ? "0.95rem" : "0.8rem",
      }}
    >
      <SeverityIcon severity={severity} width={size === "md" ? 16 : 14} height={size === "md" ? 16 : 14} />
      {label}
    </span>
  );
}
