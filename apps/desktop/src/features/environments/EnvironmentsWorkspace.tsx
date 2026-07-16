import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import type {
  ConfigurationKey,
  ConfigurationOccurrence,
  Environment,
  FindingRecord,
} from "@rayvan/core";
import type {
  ConfigurationMatrixViewModel,
  EnvironmentConfigurationStatusViewModel,
} from "@rayvan/config-engine";
import type {
  DiscoveredResourceRecord,
  EnvironmentMappingSuggestionRecord,
  ResourceBindingRecord,
} from "@rayvan/local-database";

import type { EnvironmentSyncState } from "../../lib/environments/index.js";
import { ConfigurationCellDetail } from "./ConfigurationCellDetail.js";
import { ConfigurationKeyDetail } from "./ConfigurationKeyDetail.js";
import { ConfigurationMatrix } from "./ConfigurationMatrix.js";
import {
  CreateEnvironmentDialog,
  type CreateEnvironmentSubmission,
} from "./CreateEnvironmentDialog.js";
import { EnvironmentDetail } from "./EnvironmentDetail.js";
import { EnvironmentEmptyState } from "./EnvironmentEmptyState.js";
import { EnvironmentResources } from "./EnvironmentResources.js";
import { useEnvironmentsGateway } from "./EnvironmentsContext.js";
import { EnvironmentsOverview } from "./EnvironmentsOverview.js";
import { EnvironmentsTabBar } from "./EnvironmentsTabBar.js";
import {
  mapComparisonSummary,
  mapEnvironmentToCardViewModel,
  mapResourcesViewModel,
  mapSuggestionsToViewModels,
} from "./mappers.js";
import { tabId, tabKey as tabKeyOf, tabPanelId } from "./tab-ids.js";
import type {
  ConfigurationCellSelection,
  EnvironmentCardActionId,
  EnvironmentTab,
} from "./view-models.js";

const bannerStyle: CSSProperties = {
  marginBottom: "1rem",
  padding: "0.65rem 1rem",
  borderRadius: "8px",
  border: "1px solid var(--color-border-strong)",
  background: "var(--color-surface-muted)",
};

const FIXED_TABS: EnvironmentTab[] = [
  { kind: "overview" },
  { kind: "matrix" },
  { kind: "resources" },
];

const OVERVIEW_KEY = "overview";

interface EnvironmentsWorkspaceProps {
  projectId: string | null;
}

export function EnvironmentsWorkspace({ projectId }: EnvironmentsWorkspaceProps) {
  const gateway = useEnvironmentsGateway();
  const loadGenerationRef = useRef(0);

  const [environments, setEnvironments] = useState<Environment[]>([]);
  const [keys, setKeys] = useState<ConfigurationKey[]>([]);
  const [occurrences, setOccurrences] = useState<ConfigurationOccurrence[]>([]);
  const [resources, setResources] = useState<DiscoveredResourceRecord[]>([]);
  const [bindings, setBindings] = useState<ResourceBindingRecord[]>([]);
  const [suggestions, setSuggestions] = useState<EnvironmentMappingSuggestionRecord[]>([]);
  const [matrix, setMatrix] = useState<ConfigurationMatrixViewModel | null>(null);
  const [findings, setFindings] = useState<FindingRecord[]>([]);
  const [syncState, setSyncState] = useState<EnvironmentSyncState | null>(null);
  const [configStatuses, setConfigStatuses] = useState<
    Record<string, EnvironmentConfigurationStatusViewModel>
  >({});

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [banner, setBanner] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [cellSelection, setCellSelection] = useState<ConfigurationCellSelection | null>(null);

  const [tabs, setTabs] = useState<EnvironmentTab[]>(FIXED_TABS);
  const [activeTabKey, setActiveTabKey] = useState<string>(OVERVIEW_KEY);

  const refreshForProject = useCallback(
    async (activeProjectId: string, generation: number) => {
      setLoading(true);
      setError(null);
      try {
        const [
          nextEnvironments,
          nextKeys,
          nextOccurrences,
          nextResources,
          nextBindings,
          nextSuggestions,
          nextMatrix,
          nextFindings,
          nextSyncState,
        ] = await Promise.all([
          gateway.listEnvironments(activeProjectId),
          gateway.listConfigurationKeys(activeProjectId),
          gateway.listOccurrences(activeProjectId),
          gateway.listDiscoveredResources(activeProjectId),
          gateway.listBindings(activeProjectId),
          gateway.listPendingSuggestions(activeProjectId),
          gateway.getMatrix(activeProjectId),
          gateway.listOpenFindings(activeProjectId),
          gateway.getSyncState(activeProjectId),
        ]);
        if (loadGenerationRef.current !== generation) {
          return;
        }
        setEnvironments(nextEnvironments);
        setKeys(nextKeys);
        setOccurrences(nextOccurrences);
        setResources(nextResources);
        setBindings(nextBindings);
        setSuggestions(nextSuggestions);
        setMatrix(nextMatrix);
        setFindings(nextFindings);
        setSyncState(nextSyncState);

        const statusEntries = await Promise.all(
          nextEnvironments.map(async (environment) => {
            const status = await gateway.getEnvironmentConfigurationStatus(
              activeProjectId,
              environment.id,
            );
            return [environment.id, status] as const;
          }),
        );
        if (loadGenerationRef.current !== generation) {
          return;
        }
        setConfigStatuses(Object.fromEntries(statusEntries));
      } catch (refreshError) {
        if (loadGenerationRef.current !== generation) {
          return;
        }
        setError(
          refreshError instanceof Error
            ? refreshError.message
            : "Failed to load environments.",
        );
      } finally {
        if (loadGenerationRef.current === generation) {
          setLoading(false);
        }
      }
    },
    [gateway],
  );

  useEffect(() => {
    const generation = ++loadGenerationRef.current;
    setTabs(FIXED_TABS);
    setActiveTabKey(OVERVIEW_KEY);
    setDialogOpen(false);
    setBanner(null);
    setCellSelection(null);
    setError(null);

    if (!projectId) {
      setEnvironments([]);
      setKeys([]);
      setOccurrences([]);
      setResources([]);
      setBindings([]);
      setSuggestions([]);
      setMatrix(null);
      setFindings([]);
      setSyncState(null);
      setConfigStatuses({});
      setLoading(false);
      return;
    }

    const activeProjectId = projectId;
    let cancelled = false;

    async function seedAndLoad() {
      setLoading(true);
      try {
        // Must fully await seed (including any in-flight concurrent seed)
        // before listing — otherwise Overview can render empty while fixtures
        // already exist in the gateway.
        await gateway.ensureProjectSeeded(activeProjectId);
        if (cancelled || loadGenerationRef.current !== generation) {
          return;
        }
        await refreshForProject(activeProjectId, generation);
      } catch (seedError) {
        if (cancelled || loadGenerationRef.current !== generation) {
          return;
        }
        setError(
          seedError instanceof Error
            ? seedError.message
            : "Failed to seed environments.",
        );
        setLoading(false);
      }
    }
    void seedAndLoad();

    return () => {
      cancelled = true;
    };
  }, [projectId, gateway, refreshForProject]);

  const refresh = useCallback(async () => {
    if (!projectId) {
      return;
    }
    await refreshForProject(projectId, loadGenerationRef.current);
  }, [projectId, refreshForProject]);

  const cards = useMemo(
    () =>
      environments.map((environment) =>
        mapEnvironmentToCardViewModel({
          environment,
          bindings,
          resources,
          findings,
          matrix,
          syncState,
          keyCount: keys.length,
          configStatus: configStatuses[environment.id] ?? null,
        }),
      ),
    [
      environments,
      bindings,
      resources,
      findings,
      matrix,
      syncState,
      keys.length,
      configStatuses,
    ],
  );

  const comparison = useMemo(() => mapComparisonSummary(matrix), [matrix]);
  const suggestionViewModels = useMemo(
    () => mapSuggestionsToViewModels(suggestions, resources),
    [suggestions, resources],
  );
  const resourcesViewModel = useMemo(
    () => mapResourcesViewModel({ environments, resources, bindings }),
    [environments, resources, bindings],
  );

  const showLocalOnlyBanner =
    environments.length > 0 &&
    (environments.every((environment) => environment.status === "local_only") ||
      bindings.filter((binding) => binding.bindingStatus === "active").length === 0);

  const openEnvironmentTab = useCallback((environmentId: string, label: string) => {
    setTabs((current) => {
      if (
        current.some(
          (tab) => tab.kind === "environment" && tab.environmentId === environmentId,
        )
      ) {
        return current;
      }
      return [...current, { kind: "environment", environmentId, label }];
    });
    setActiveTabKey(`environment:${environmentId}`);
  }, []);

  const openKeyTab = useCallback((configurationKeyId: string, label: string) => {
    setTabs((current) => {
      if (
        current.some(
          (tab) =>
            tab.kind === "configurationKey" &&
            tab.configurationKeyId === configurationKeyId,
        )
      ) {
        return current;
      }
      return [...current, { kind: "configurationKey", configurationKeyId, label }];
    });
    setActiveTabKey(`key:${configurationKeyId}`);
  }, []);

  const closeTab = useCallback((key: string) => {
    setTabs((current) => current.filter((tab) => tabKeyOf(tab) !== key));
    setActiveTabKey((current) => (current === key ? OVERVIEW_KEY : current));
  }, []);

  const handleCardAction = useCallback(
    (environmentId: string, actionId: EnvironmentCardActionId) => {
      const environment = environments.find((item) => item.id === environmentId);
      if (
        actionId === "open" ||
        actionId === "edit" ||
        actionId === "review_changes"
      ) {
        openEnvironmentTab(environmentId, environment?.name ?? "Environment");
        return;
      }
      if (actionId === "sync") {
        if (!projectId) {
          return;
        }
        void gateway
          .syncWithIntegrations(projectId, { environmentId })
          .then(() => refresh())
          .then(() => setBanner(`Synced ${environment?.name ?? "environment"} (read-only).`))
          .catch((syncError: unknown) => {
            setError(syncError instanceof Error ? syncError.message : "Sync failed.");
          });
        return;
      }
      if (actionId === "archive") {
        void gateway
          .archiveEnvironment(environmentId)
          .then(() => refresh())
          .then(() => {
            closeTab(`environment:${environmentId}`);
            setBanner(`${environment?.name ?? "Environment"} archived.`);
          })
          .catch((archiveError: unknown) => {
            setError(
              archiveError instanceof Error
                ? archiveError.message
                : "Failed to archive environment.",
            );
          });
      }
    },
    [closeTab, environments, gateway, openEnvironmentTab, projectId, refresh],
  );

  const handleCreate = useCallback(
    async (submission: CreateEnvironmentSubmission) => {
      if (!projectId) {
        throw new Error("Select a project before creating an environment.");
      }
      const created = await gateway.createEnvironment({
        projectId,
        name: submission.name,
        kind: submission.kind,
        description: submission.description,
        presentation: submission.presentation,
      });
      await refresh();
      setBanner(
        `${created.name} created as local only. Sync with integrations when you are ready.`,
      );
      openEnvironmentTab(created.id, created.name);
    },
    [gateway, openEnvironmentTab, projectId, refresh],
  );

  const handleSyncAll = useCallback(() => {
    if (!projectId) {
      return;
    }
    void gateway
      .syncWithIntegrations(projectId)
      .then((result) => {
        setBanner(
          result.phase === "failed"
            ? "Sync finished with partial failures (read-only)."
            : "Sync complete (read-only).",
        );
        return refresh();
      })
      .catch((syncError: unknown) => {
        setError(syncError instanceof Error ? syncError.message : "Sync failed.");
      });
  }, [gateway, projectId, refresh]);

  const handleCancelSync = useCallback(() => {
    if (!projectId) {
      return;
    }
    void gateway.cancelSync(projectId).then(() => refresh());
  }, [gateway, projectId, refresh]);

  const handleAcceptSuggestion = useCallback(
    (suggestionId: string) => {
      const suggestion = suggestions.find((item) => item.id === suggestionId);
      void gateway
        .acceptSuggestion({
          suggestionId,
          environmentId: suggestion?.suggestedEnvironmentId,
        })
        .then(() => refresh())
        .then(() => setBanner("Mapping suggestion accepted and resource bound."))
        .catch((acceptError: unknown) => {
          setError(
            acceptError instanceof Error
              ? acceptError.message
              : "Failed to accept suggestion.",
          );
        });
    },
    [gateway, refresh, suggestions],
  );

  const handleRejectSuggestion = useCallback(
    (suggestionId: string) => {
      void gateway
        .rejectSuggestion(suggestionId)
        .then(() => refresh())
        .then(() => setBanner("Mapping suggestion rejected."))
        .catch((rejectError: unknown) => {
          setError(
            rejectError instanceof Error
              ? rejectError.message
              : "Failed to reject suggestion.",
          );
        });
    },
    [gateway, refresh],
  );

  if (!projectId) {
    return <EnvironmentEmptyState variant="no-project" />;
  }

  const detailEnvironmentTabs = tabs.filter(
    (tab): tab is Extract<EnvironmentTab, { kind: "environment" }> =>
      tab.kind === "environment",
  );
  const detailKeyTabs = tabs.filter(
    (tab): tab is Extract<EnvironmentTab, { kind: "configurationKey" }> =>
      tab.kind === "configurationKey",
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

      <EnvironmentsTabBar
        tabs={tabs}
        activeTabKey={activeTabKey}
        onSelect={setActiveTabKey}
        onClose={closeTab}
      />

      <div
        role="tabpanel"
        id={tabPanelId("overview")}
        aria-labelledby={tabId("overview")}
        hidden={activeTabKey !== "overview"}
      >
        <EnvironmentsOverview
          cards={cards}
          comparison={comparison}
          suggestions={suggestionViewModels}
          syncState={syncState}
          showLocalOnlyBanner={showLocalOnlyBanner}
          canMutate={projectId !== null}
          onCreateEnvironment={() => setDialogOpen(true)}
          onOpenMatrix={() => setActiveTabKey("matrix")}
          onCardAction={handleCardAction}
          onAcceptSuggestion={handleAcceptSuggestion}
          onRejectSuggestion={handleRejectSuggestion}
          onSync={handleSyncAll}
          onCancelSync={handleCancelSync}
        />
        {loading && environments.length === 0 ? (
          <p style={{ color: "var(--color-text-secondary)" }}>Loading environments…</p>
        ) : null}
      </div>

      <div
        role="tabpanel"
        id={tabPanelId("matrix")}
        aria-labelledby={tabId("matrix")}
        hidden={activeTabKey !== "matrix"}
      >
        <ConfigurationMatrix
          matrix={matrix}
          onSelectCell={setCellSelection}
          onOpenKey={openKeyTab}
          onOpenEnvironment={openEnvironmentTab}
        />
      </div>

      <div
        role="tabpanel"
        id={tabPanelId("resources")}
        aria-labelledby={tabId("resources")}
        hidden={activeTabKey !== "resources"}
      >
        <EnvironmentResources
          viewModel={resourcesViewModel}
          environments={environments}
          onAttach={(discoveredResourceId, environmentId) => {
            void gateway
              .attachResource({
                projectId,
                environmentId,
                discoveredResourceId,
              })
              .then(() => refresh())
              .catch((attachError: unknown) => {
                setError(
                  attachError instanceof Error
                    ? attachError.message
                    : "Failed to attach resource.",
                );
              });
          }}
          onMove={(bindingId, environmentId) => {
            void gateway
              .moveResource({ bindingId, environmentId })
              .then(() => refresh())
              .catch((moveError: unknown) => {
                setError(
                  moveError instanceof Error ? moveError.message : "Failed to move resource.",
                );
              });
          }}
          onDetach={(bindingId) => {
            void gateway
              .detachResource(bindingId)
              .then(() => refresh())
              .catch((detachError: unknown) => {
                setError(
                  detachError instanceof Error
                    ? detachError.message
                    : "Failed to detach resource.",
                );
              });
          }}
        />
      </div>

      {detailEnvironmentTabs.map((tab) => {
        const environment = environments.find((item) => item.id === tab.environmentId);
        const key = tabKeyOf(tab);
        const card = cards.find((item) => item.environmentId === tab.environmentId);
        const envResources =
          resourcesViewModel.groups.find((group) => group.environmentId === tab.environmentId)
            ?.items ?? [];
        return (
          <div
            key={key}
            role="tabpanel"
            id={tabPanelId(key)}
            aria-labelledby={tabId(key)}
            hidden={activeTabKey !== key}
          >
            {environment && projectId ? (
              <EnvironmentDetail
                projectId={projectId}
                environment={environment}
                card={card}
                resources={envResources}
                findings={findings}
                keys={keys}
                occurrences={occurrences}
                gateway={gateway}
                onSync={(environmentId) => handleCardAction(environmentId, "sync")}
                onArchive={(environmentId) => handleCardAction(environmentId, "archive")}
                onOpenMatrix={() => setActiveTabKey("matrix")}
                onRefresh={refresh}
                onBanner={setBanner}
              />
            ) : null}
          </div>
        );
      })}

      {detailKeyTabs.map((tab) => {
        const configurationKey = keys.find((item) => item.id === tab.configurationKeyId);
        const key = tabKeyOf(tab);
        return (
          <div
            key={key}
            role="tabpanel"
            id={tabPanelId(key)}
            aria-labelledby={tabId(key)}
            hidden={activeTabKey !== key}
          >
            {configurationKey ? (
              <ConfigurationKeyDetail
                configurationKey={configurationKey}
                environments={environments}
                occurrences={occurrences}
                matrix={matrix}
                onOpenEnvironment={openEnvironmentTab}
              />
            ) : null}
          </div>
        );
      })}

      <ConfigurationCellDetail
        selection={cellSelection}
        onClose={() => setCellSelection(null)}
        onOpenKey={openKeyTab}
        onOpenEnvironment={openEnvironmentTab}
      />

      <CreateEnvironmentDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        onSubmit={handleCreate}
      />
    </section>
  );
}
