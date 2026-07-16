import type { ComponentType } from "react";

import { ActionsPage } from "../features/actions/ActionsPage.js";
import { AgentMcpPage } from "../features/agent-mcp/AgentMcpPage.js";
import { DeploymentsPage } from "../features/deployments/DeploymentsPage.js";
import { EnvironmentsPage } from "../features/environments/EnvironmentsPage.js";
import { FindingsPage } from "../features/findings/FindingsPage.js";
import { IntegrationsPage } from "../features/integrations/IntegrationsPage.js";
import { OverviewPage } from "../features/projects/OverviewPage.js";
import { SettingsPage } from "../features/settings/SettingsPage.js";
import type { AppSection } from "./navigation/sections.js";

export const SECTION_PAGES: Record<AppSection, ComponentType> = {
  overview: OverviewPage,
  environments: EnvironmentsPage,
  integrations: IntegrationsPage,
  deployments: DeploymentsPage,
  findings: FindingsPage,
  actions: ActionsPage,
  "agent-mcp": AgentMcpPage,
  settings: SettingsPage,
};
