import type { ReactElement, ReactNode, SVGProps } from "react";

import type { AppSection } from "../navigation/sections.js";

type IconProps = SVGProps<SVGSVGElement>;

function Icon({ children, ...props }: IconProps & { children: ReactNode }) {
  return (
    <svg
      width="16"
      height="16"
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

function OverviewIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <>
        <rect x="3" y="3" width="7" height="9" rx="1" />
        <rect x="14" y="3" width="7" height="5" rx="1" />
        <rect x="14" y="12" width="7" height="9" rx="1" />
        <rect x="3" y="16" width="7" height="5" rx="1" />
      </>
    </Icon>
  );
}

function EnvironmentsIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <>
        <path d="M12 2 2 7l10 5 10-5-10-5Z" />
        <path d="m2 17 10 5 10-5" />
        <path d="m2 12 10 5 10-5" />
      </>
    </Icon>
  );
}

function IntegrationsIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <>
        <path d="M12 2v6" />
        <path d="M12 16v6" />
        <path d="M2 12h6" />
        <path d="M16 12h6" />
        <circle cx="12" cy="12" r="3" />
      </>
    </Icon>
  );
}

function DeploymentsIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <>
        <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09Z" />
        <path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2Z" />
        <path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0" />
        <path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5" />
      </>
    </Icon>
  );
}

function FindingsIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <>
        <path d="m21 21-4.3-4.3" />
        <circle cx="11" cy="11" r="8" />
      </>
    </Icon>
  );
}

function ActionsIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </Icon>
  );
}

function AgentMcpIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <>
        <path d="M12 8V4H8" />
        <rect width="16" height="12" x="4" y="8" rx="2" />
        <path d="M2 14h2" />
        <path d="M20 14h2" />
        <path d="M15 13v2" />
        <path d="M9 13v2" />
      </>
    </Icon>
  );
}

function SettingsIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <>
        <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2Z" />
        <circle cx="12" cy="12" r="3" />
      </>
    </Icon>
  );
}

const SECTION_ICONS: Record<AppSection, (props: IconProps) => ReactElement> = {
  overview: OverviewIcon,
  environments: EnvironmentsIcon,
  integrations: IntegrationsIcon,
  deployments: DeploymentsIcon,
  findings: FindingsIcon,
  actions: ActionsIcon,
  "agent-mcp": AgentMcpIcon,
  settings: SettingsIcon,
};

export function SectionIcon({
  section,
  ...props
}: IconProps & { section: AppSection }) {
  const Component = SECTION_ICONS[section];
  return <Component {...props} />;
}
