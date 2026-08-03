import type { CSSProperties } from "react";

import { Button } from "@rayvan/ui";



import { IntegrationCardGrid } from "./IntegrationCardGrid.js";

import { IntegrationEmptyState } from "./IntegrationEmptyState.js";

import { IntegrationIcon } from "./icons.js";

import { resolveIntegrationTheme } from "./theme.js";

import type {

  IntegrationCardActionId,

  LibraryPluginViewModel,

  PluginIntegrationCardViewModel,

} from "./view-models.js";



const headerStyle: CSSProperties = {

  display: "flex",

  justifyContent: "space-between",

  alignItems: "flex-start",

  gap: "1rem",

  marginBottom: "1.25rem",

};



const availableListStyle: CSSProperties = {

  display: "flex",

  flexDirection: "column",

  gap: "0.75rem",

  marginTop: "1.5rem",

};



const availableItemStyle: CSSProperties = {

  display: "flex",

  gap: "0.75rem",

  alignItems: "center",

  padding: "0.85rem 1rem",

  borderRadius: "8px",

  border: "1px dashed var(--color-border-strong)",

  background: "var(--color-surface-muted)",

};



interface IntegrationsHomeProps {

  cards: PluginIntegrationCardViewModel[];

  /** Installed plugins not yet connected to this project. */

  availableToSetup: LibraryPluginViewModel[];

  canAddIntegration: boolean;

  onOpen: (connectionId: string) => void;

  onAction: (connectionId: string, actionId: IntegrationCardActionId) => void;

  onAddIntegration: () => void;

  onSetupPlugin: (installedPluginId: string) => void;

}



export function IntegrationsHome({

  cards,

  availableToSetup,

  canAddIntegration,

  onOpen,

  onAction,

  onAddIntegration,

  onSetupPlugin,

}: IntegrationsHomeProps) {

  return (

    <section>

      <div style={headerStyle}>

        <div>

          <h2 style={{ marginTop: 0, marginBottom: "0.35rem" }}>Integrations</h2>

          <p style={{ margin: 0, color: "var(--color-text-secondary)" }}>

            Connect Rayvan to the services used by this project.

          </p>

        </div>

        <Button onClick={onAddIntegration} disabled={!canAddIntegration}>

          + Add integration

        </Button>

      </div>



      {cards.length === 0 ? (

        <IntegrationEmptyState

          onAddIntegration={canAddIntegration ? onAddIntegration : undefined}

        />

      ) : (

        <IntegrationCardGrid cards={cards} onOpen={onOpen} onAction={onAction} />

      )}



      {availableToSetup.length > 0 ? (

        <div style={availableListStyle}>

          <h3 style={{ margin: 0, fontSize: "1rem" }}>Installed — needs setup</h3>

          <p style={{ margin: 0, color: "var(--color-text-secondary)" }}>

            These plugins are installed on this machine but not connected to

            this project yet.

          </p>

          {availableToSetup.map((plugin) => {

            const theme = resolveIntegrationTheme(plugin.theme);

            return (

              <div key={plugin.installedPluginId} style={availableItemStyle}>

                <IntegrationIcon icon={plugin.icon} theme={theme} />

                <div style={{ flex: 1, minWidth: 0 }}>

                  <strong>{plugin.name}</strong>

                  <div style={{ fontSize: "0.8rem", color: "var(--color-text-muted)" }}>

                    {plugin.publisher} · v{plugin.version}

                    {plugin.badge === "unsigned-dev"

                      ? " · Unsigned (development)"

                      : null}

                  </div>

                </div>

                <Button

                  onClick={() => onSetupPlugin(plugin.installedPluginId)}

                  disabled={!canAddIntegration}

                >

                  Set up

                </Button>

              </div>

            );

          })}

        </div>

      ) : null}

    </section>

  );

}


