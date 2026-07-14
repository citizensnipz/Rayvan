import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import type { PluginExecutionActor } from "@rayvan/plugin-sdk";
import type {
  InstalledPluginRecord,
  PluginConnectionRecord,
  PluginPermissionGrantRecord,
} from "@rayvan/local-database";

import { AddIntegrationDialog } from "./AddIntegrationDialog.js";
import { IntegrationDetail } from "./IntegrationDetail.js";
import { useIntegrationsGateway } from "./IntegrationsContext.js";
import { IntegrationsHome } from "./IntegrationsHome.js";
import { IntegrationsTabBar } from "./IntegrationsTabBar.js";
import type { AddIntegrationSubmission } from "./InstalledPluginLibrary.js";
import {
  mapConnectionToCardViewModel,
  mapConnectionToDetailViewModel,
  mapInstalledPluginToLibraryViewModel,
} from "./mappers.js";
import { tabId, tabKey as tabKeyOf, tabPanelId } from "./tab-ids.js";
import type {
  IntegrationCardActionId,
  IntegrationTab,
  PluginIntegrationDetailViewModel,
} from "./view-models.js";

const DESKTOP_USER_ACTOR: PluginExecutionActor = {
  type: "user",
  id: "desktop-user",
  displayName: "You",
};

const bannerStyle: CSSProperties = {
  marginBottom: "1rem",
  padding: "0.65rem 1rem",
  borderRadius: "8px",
  border: "1px solid var(--color-border-strong)",
  background: "var(--color-surface-muted)",
};

const HOME_TAB_KEY = "home";

interface IntegrationsWorkspaceProps {
  projectId: string | null;
}

export function IntegrationsWorkspace({ projectId }: IntegrationsWorkspaceProps) {
  const gateway = useIntegrationsGateway();
  const loadGenerationRef = useRef(0);

  const [installedPlugins, setInstalledPlugins] = useState<InstalledPluginRecord[]>([]);
  const [connections, setConnections] = useState<PluginConnectionRecord[]>([]);
  const [grantsByConnectionId, setGrantsByConnectionId] = useState<
    Record<string, PluginPermissionGrantRecord[]>
  >({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [tabs, setTabs] = useState<IntegrationTab[]>([{ kind: "home" }]);
  const [activeTabKey, setActiveTabKey] = useState<string>(HOME_TAB_KEY);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [banner, setBanner] = useState<string | null>(null);

  const refreshForProject = useCallback(
    async (activeProjectId: string, generation: number) => {
      setLoading(true);
      setError(null);
      try {
        const [nextInstalled, nextConnections] = await Promise.all([
          gateway.listInstalledPlugins(),
          gateway.listConnectionsByProject(activeProjectId),
        ]);
        if (loadGenerationRef.current !== generation) {
          return;
        }
        setInstalledPlugins(nextInstalled);
        setConnections(nextConnections);
      } catch (refreshError) {
        if (loadGenerationRef.current !== generation) {
          return;
        }
        setError(
          refreshError instanceof Error
            ? refreshError.message
            : "Failed to load integrations.",
        );
      } finally {
        if (loadGenerationRef.current === generation) {
          setLoading(false);
        }
      }
    },
    [gateway],
  );

  // Reset workspace tabs whenever the current project changes, then seed
  // (idempotent) and load data for the new project.
  useEffect(() => {
    const generation = ++loadGenerationRef.current;
    setTabs([{ kind: "home" }]);
    setActiveTabKey(HOME_TAB_KEY);
    setDialogOpen(false);
    setBanner(null);
    setGrantsByConnectionId({});
    setError(null);

    if (!projectId) {
      setInstalledPlugins([]);
      setConnections([]);
      setLoading(false);
      return;
    }

    const activeProjectId = projectId;
    async function seedAndLoad() {
      try {
        await gateway.ensureProjectSeeded(activeProjectId);
      } catch (seedError) {
        if (loadGenerationRef.current === generation) {
          setError(
            seedError instanceof Error ? seedError.message : "Failed to seed integrations.",
          );
        }
        return;
      }
      if (loadGenerationRef.current === generation) {
        await refreshForProject(activeProjectId, generation);
      }
    }
    void seedAndLoad();
  }, [projectId, gateway, refreshForProject]);

  const refresh = useCallback(async () => {
    if (!projectId) {
      setInstalledPlugins([]);
      setConnections([]);
      return;
    }
    await refreshForProject(projectId, loadGenerationRef.current);
  }, [projectId, refreshForProject]);

  const installedByPluginId = useMemo(() => {
    const map = new Map<string, InstalledPluginRecord>();
    for (const installed of installedPlugins) {
      map.set(installed.id, installed);
    }
    return map;
  }, [installedPlugins]);

  const cards = useMemo(
    () =>
      connections
        .map((connection) => {
          const installed = installedByPluginId.get(connection.installedPluginId);
          return installed ? mapConnectionToCardViewModel(connection, installed) : null;
        })
        .filter((card): card is NonNullable<typeof card> => card !== null),
    [connections, installedByPluginId],
  );

  const libraryPlugins = useMemo(
    () =>
      installedPlugins
        .map((installed) => mapInstalledPluginToLibraryViewModel(installed, connections))
        .filter((plugin) => plugin.eligible),
    [installedPlugins, connections],
  );

  const loadGrants = useCallback(
    async (connectionId: string) => {
      const grants = await gateway.listPermissionGrants(connectionId);
      setGrantsByConnectionId((current) => ({ ...current, [connectionId]: grants }));
    },
    [gateway],
  );

  const openTab = useCallback(
    (connectionId: string, labelOverride?: string) => {
      setTabs((current) => {
        if (current.some((tab) => tab.kind === "detail" && tab.connectionId === connectionId)) {
          return current;
        }
        const connection = connections.find((item) => item.id === connectionId);
        return [
          ...current,
          {
            kind: "detail",
            connectionId,
            label: labelOverride ?? connection?.name ?? "Integration",
          },
        ];
      });
      setActiveTabKey(connectionId);
      void loadGrants(connectionId);
    },
    [connections, loadGrants],
  );

  const closeTab = useCallback((connectionId: string) => {
    setTabs((current) => current.filter((tab) => tabKeyOf(tab) !== connectionId));
    setActiveTabKey((current) => (current === connectionId ? HOME_TAB_KEY : current));
  }, []);

  const handleCardAction = useCallback(
    (connectionId: string, actionId: IntegrationCardActionId) => {
      if (actionId === "open" || actionId === "configure") {
        openTab(connectionId);
        return;
      }
      if (actionId === "sync") {
        void gateway
          .markConnected(connectionId)
          .then(() => refresh())
          .catch((syncError: unknown) => {
            setError(syncError instanceof Error ? syncError.message : "Sync failed.");
          });
      }
    },
    [gateway, openTab, refresh],
  );

  const handleAddSubmission = useCallback(
    async (submission: AddIntegrationSubmission) => {
      if (!projectId) {
        throw new Error("Select a project before adding an integration.");
      }
      const installed = installedByPluginId.get(submission.installedPluginId);
      if (!installed) {
        throw new Error("Selected plugin is not installed.");
      }

      const connection = await gateway.createConnection({
        installedPluginId: submission.installedPluginId,
        projectId,
        name: submission.connectionName,
      });

      if (submission.permissions.length > 0) {
        await gateway.grantPermissions({
          pluginId: installed.pluginId,
          connectionId: connection.id,
          projectId,
          permissions: submission.permissions,
          grantedBy: DESKTOP_USER_ACTOR,
        });
      }

      await refresh();
      setDialogOpen(false);
      setBanner(`${submission.connectionName} was connected successfully.`);
      openTab(connection.id, connection.name);
    },
    [gateway, installedByPluginId, openTab, projectId, refresh],
  );

  const handleDisconnect = useCallback(
    (connectionId: string) => {
      void gateway
        .disconnectConnection(connectionId)
        .then(() => Promise.all([refresh(), loadGrants(connectionId)]))
        .catch((disconnectError: unknown) => {
          setError(
            disconnectError instanceof Error ? disconnectError.message : "Failed to disconnect.",
          );
        });
    },
    [gateway, loadGrants, refresh],
  );

  const handleReconnect = useCallback(
    (connectionId: string) => {
      void gateway
        .markConnected(connectionId)
        .then(() => refresh())
        .catch((reconnectError: unknown) => {
          setError(
            reconnectError instanceof Error ? reconnectError.message : "Failed to reconnect.",
          );
        });
    },
    [gateway, refresh],
  );

  function detailViewModelFor(connectionId: string): PluginIntegrationDetailViewModel | null {
    const connection = connections.find((item) => item.id === connectionId);
    const installed = connection
      ? installedByPluginId.get(connection.installedPluginId)
      : undefined;
    if (!connection || !installed) {
      return null;
    }
    const grants = grantsByConnectionId[connectionId] ?? [];
    return mapConnectionToDetailViewModel(connection, installed, grants);
  }

  const detailTabs = tabs.filter(
    (tab): tab is Extract<IntegrationTab, { kind: "detail" }> => tab.kind === "detail",
  );

  return (
    <section>
      {banner ? (
        <div role="status" style={bannerStyle}>
          {banner}
        </div>
      ) : null}

      {error ? (
        <div role="alert" style={{ ...bannerStyle, borderColor: "var(--color-danger)" }}>
          {error}
        </div>
      ) : null}

      <IntegrationsTabBar
        tabs={tabs}
        activeTabKey={activeTabKey}
        onSelect={setActiveTabKey}
        onClose={closeTab}
      />

      <div
        role="tabpanel"
        id={tabPanelId(HOME_TAB_KEY)}
        aria-labelledby={tabId(HOME_TAB_KEY)}
        hidden={activeTabKey !== HOME_TAB_KEY}
      >
        <IntegrationsHome
          cards={cards}
          canAddIntegration={projectId !== null}
          onOpen={openTab}
          onAction={handleCardAction}
          onAddIntegration={() => setDialogOpen(true)}
        />
        {loading && cards.length === 0 ? (
          <p style={{ color: "var(--color-text-secondary)" }}>Loading integrations&hellip;</p>
        ) : null}
      </div>

      {detailTabs.map((tab) => {
        const detail = detailViewModelFor(tab.connectionId);
        const key = tab.connectionId;
        return (
          <div
            key={key}
            role="tabpanel"
            id={tabPanelId(key)}
            aria-labelledby={tabId(key)}
            hidden={activeTabKey !== key}
          >
            {detail ? (
              <IntegrationDetail
                detail={detail}
                onDisconnect={handleDisconnect}
                onReconnect={handleReconnect}
              />
            ) : null}
          </div>
        );
      })}

      <AddIntegrationDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        plugins={libraryPlugins}
        onSubmit={handleAddSubmission}
      />
    </section>
  );
}
