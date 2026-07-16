import type { CSSProperties } from "react";
import { NavItem } from "@rayvan/ui";

import { useResolvedCurrentProject } from "../CurrentProjectContext.js";
import { useAppNavigation } from "../navigation/AppNavigationContext.js";
import { APP_SECTION_DEFINITIONS } from "../navigation/sections.js";
import { SectionIcon } from "./sidebar-icons.js";

const asideStyle: CSSProperties = {
  borderRight: "1px solid var(--color-border)",
  background: "var(--color-surface)",
  padding: "1rem 0.75rem",
  display: "flex",
  flexDirection: "column",
  gap: "0.25rem",
  minHeight: 0,
  overflow: "auto",
};

const itemContentStyle: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: "0.6rem",
};

export function Sidebar() {
  const { hasProject } = useResolvedCurrentProject();
  const { activeSection, setActiveSection } = useAppNavigation();

  return (
    <aside style={asideStyle} aria-label="Primary">
      <nav aria-label="Main">
        <ul
          style={{
            listStyle: "none",
            margin: 0,
            padding: 0,
            display: "grid",
            gap: "0.15rem",
          }}
        >
          {APP_SECTION_DEFINITIONS.map((section) => {
            const disabled = section.requiresProject && !hasProject;
            return (
              <li key={section.id}>
                <NavItem
                  active={activeSection === section.id}
                  disabled={disabled}
                  onClick={() => setActiveSection(section.id)}
                >
                  <span style={itemContentStyle}>
                    <SectionIcon section={section.id} />
                    {section.label}
                  </span>
                </NavItem>
              </li>
            );
          })}
        </ul>
      </nav>
    </aside>
  );
}
