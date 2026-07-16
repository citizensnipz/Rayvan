import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import type {
  FindingLifecycleEventRecord,
  FindingRecord,
  FindingSummary,
} from "@rayvan/core";
import { Button } from "@rayvan/ui";

import type {
  FindingEvaluationState,
  FindingsDismissReason,
  FindingsSuppressPreset,
} from "../../lib/findings/index.js";
import { FindingDetail } from "./FindingDetail.js";
import { FindingEmptyState } from "./FindingEmptyState.js";
import { FindingFilters } from "./FindingFilters.js";
import { useFindingsGateway } from "./FindingsContext.js";
import { FindingsList } from "./FindingsList.js";
import { FindingsTabBar } from "./FindingsTabBar.js";
import {
  filterFindings,
  groupFindingsBySeverity,
  mapFindingDetail,
  mapFindingToListItem,
} from "./mappers.js";
import { tabId, tabKey as tabKeyOf, tabPanelId } from "./tab-ids.js";
import {
  defaultFindingFilters,
  mapHeaderSummary,
  type FindingFiltersState,
  type FindingsTab,
} from "./view-models.js";

const bannerStyle: CSSProperties = {
  marginBottom: "1rem",
  padding: "0.65rem 1rem",
  borderRadius: "8px",
  border: "1px solid var(--color-border-strong)",
  background: "var(--color-surface-muted)",
};

const LIST_KEY = "list";
const FIXED_TABS: FindingsTab[] = [{ kind: "list" }];

interface FindingsWorkspaceProps {
  projectId: string | null;
}

export function FindingsWorkspace({ projectId }: FindingsWorkspaceProps) {
  const gateway = useFindingsGateway();
  const loadGenerationRef = useRef(0);

  const [findings, setFindings] = useState<FindingRecord[]>([]);
  const [summary, setSummary] = useState<FindingSummary | null>(null);
  const [evaluationState, setEvaluationState] =
    useState<FindingEvaluationState | null>(null);
  const [lifecycleById, setLifecycleById] = useState<
    Record<string, FindingLifecycleEventRecord[]>
  >({});
  const [loading, setLoading] = useState(false);
  const [actionBusy, setActionBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [banner, setBanner] = useState<string | null>(null);
  const [filters, setFilters] = useState<FindingFiltersState>(defaultFindingFilters);
  const [tabs, setTabs] = useState<FindingsTab[]>(FIXED_TABS);
  const [activeTabKey, setActiveTabKey] = useState(LIST_KEY);
  const [hasSeeded, setHasSeeded] = useState(false);

  const refreshForProject = useCallback(
    async (activeProjectId: string, generation: number) => {
      setLoading(true);
      setError(null);
      try {
        const [nextFindings, nextSummary, nextEvaluation] = await Promise.all([
          gateway.listFindings({
            projectId: activeProjectId,
            includeResolved: true,
          }),
          gateway.getProjectSummary(activeProjectId),
          gateway.getEvaluationState(activeProjectId),
        ]);
        if (loadGenerationRef.current !== generation) {
          return;
        }
        setFindings(nextFindings);
        setSummary(nextSummary);
        setEvaluationState(nextEvaluation);
        setHasSeeded(true);
      } catch (refreshError) {
        if (loadGenerationRef.current !== generation) {
          return;
        }
        setError(
          refreshError instanceof Error
            ? refreshError.message
            : "Failed to load findings.",
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
    setActiveTabKey(LIST_KEY);
    setBanner(null);
    setError(null);
    setFilters(defaultFindingFilters());
    setLifecycleById({});
    setHasSeeded(false);

    if (!projectId) {
      setFindings([]);
      setSummary(null);
      setEvaluationState(null);
      setLoading(false);
      return;
    }

    const activeProjectId = projectId;
    let cancelled = false;

    async function seedAndLoad() {
      setLoading(true);
      try {
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
            : "Failed to seed findings.",
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

  const filtered = useMemo(
    () => filterFindings(findings, filters),
    [findings, filters],
  );

  const listItems = useMemo(
    () => filtered.map((record) => mapFindingToListItem(record)),
    [filtered],
  );

  const groups = useMemo(
    () => groupFindingsBySeverity(listItems),
    [listItems],
  );

  const header = useMemo(() => mapHeaderSummary(summary), [summary]);

  const environmentOptions = useMemo(() => {
    const map = new Map<string, string>();
    for (const record of findings) {
      if (record.environmentId) {
        map.set(
          record.environmentId,
          mapFindingToListItem(record).environmentLabel ?? record.environmentId,
        );
      }
    }
    return [...map.entries()].map(([id, label]) => ({ id, label }));
  }, [findings]);

  const integrationOptions = useMemo(() => {
    const map = new Map<string, string>();
    for (const record of findings) {
      if (record.connectionId) {
        map.set(
          record.connectionId,
          mapFindingToListItem(record).integrationLabel ?? record.connectionId,
        );
      }
    }
    return [...map.entries()].map(([id, label]) => ({ id, label }));
  }, [findings]);

  const openDetailTab = useCallback(async (findingId: string) => {
    const record = findings.find((item) => item.id === findingId);
    const label = record?.title ?? "Finding";
    setTabs((current) => {
      if (current.some((tab) => tab.kind === "detail" && tab.findingId === findingId)) {
        return current;
      }
      return [...current, { kind: "detail", findingId, label }];
    });
    setActiveTabKey(`detail:${findingId}`);

    if (!lifecycleById[findingId]) {
      try {
        const events = await gateway.getLifecycleEvents(findingId);
        setLifecycleById((current) => ({ ...current, [findingId]: events }));
      } catch {
        // Detail can still render without history.
      }
    }
  }, [findings, gateway, lifecycleById]);

  const closeTab = useCallback((key: string) => {
    setTabs((current) => current.filter((tab) => tabKeyOf(tab) !== key));
    setActiveTabKey((current) => (current === key ? LIST_KEY : current));
  }, []);

  const handleScan = useCallback(() => {
    if (!projectId) {
      return;
    }
    setBanner(null);
    void gateway
      .evaluateProject(projectId)
      .then(async (result) => {
        setBanner(
          result.run.status === "partially_succeeded"
            ? "Scan finished with partial failures."
            : `Scan complete — created ${result.created.length}, updated ${result.updated.length}.`,
        );
        await refresh();
      })
      .catch((scanError: unknown) => {
        setError(
          scanError instanceof Error ? scanError.message : "Scan failed.",
        );
      });
  }, [gateway, projectId, refresh]);

  const handleCancelScan = useCallback(() => {
    if (!projectId || !gateway.cancelEvaluation) {
      return;
    }
    void gateway.cancelEvaluation(projectId).then(() => refresh());
  }, [gateway, projectId, refresh]);

  const runLifecycle = useCallback(
    async (action: () => Promise<FindingRecord>, successMessage: string) => {
      setActionBusy(true);
      setError(null);
      try {
        const updated = await action();
        const events = await gateway.getLifecycleEvents(updated.id);
        setLifecycleById((current) => ({ ...current, [updated.id]: events }));
        setBanner(successMessage);
        await refresh();
      } catch (actionError) {
        setError(
          actionError instanceof Error
            ? actionError.message
            : "Finding action failed.",
        );
      } finally {
        setActionBusy(false);
      }
    },
    [gateway, refresh],
  );

  if (!projectId) {
    return <FindingEmptyState variant="no-project" />;
  }

  const detailTabs = tabs.filter(
    (tab): tab is Extract<FindingsTab, { kind: "detail" }> => tab.kind === "detail",
  );

  const emptyVariant = (() => {
    if (evaluationState?.inProgress) {
      return "scanning" as const;
    }
    if (evaluationState?.phase === "partially_succeeded") {
      return "partial-failure" as const;
    }
    if (!hasSeeded && !loading) {
      return "never-evaluated" as const;
    }
    if (findings.length === 0) {
      return "never-evaluated" as const;
    }
    if (filtered.length === 0) {
      return filters.openOnly &&
        findings.every(
          (item) => item.status !== "open" && item.status !== "acknowledged",
        )
        ? ("no-active" as const)
        : ("no-matches" as const);
    }
    return null;
  })();

  return (
    <section>
      {/*
        Dev fixtures: open Findings with a project selected.
        ensureProjectSeeded auto-seeds FindingRecords; “Scan project” runs evaluateProject.
      */}
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

      <FindingsTabBar
        tabs={tabs}
        activeTabKey={activeTabKey}
        onSelect={setActiveTabKey}
        onClose={closeTab}
      />

      <div
        role="tabpanel"
        id={tabPanelId(LIST_KEY)}
        aria-labelledby={tabId(LIST_KEY)}
        hidden={activeTabKey !== LIST_KEY}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            gap: "1rem",
            flexWrap: "wrap",
            marginBottom: "1rem",
            alignItems: "flex-start",
          }}
        >
          <div>
            <h2 style={{ margin: "0 0 0.35rem" }}>Findings</h2>
            <p
              style={{ margin: 0, color: "var(--color-text-secondary)" }}
              aria-label={header.headline}
            >
              {header.headline}
            </p>
          </div>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            <Button onClick={handleScan} disabled={evaluationState?.inProgress}>
              Scan project
            </Button>
            {evaluationState?.inProgress && gateway.cancelEvaluation ? (
              <Button onClick={handleCancelScan}>Cancel scan</Button>
            ) : null}
          </div>
        </div>

        <FindingFilters
          filters={filters}
          environments={environmentOptions}
          integrations={integrationOptions}
          onChange={setFilters}
        />

        {loading && findings.length === 0 ? (
          <p style={{ color: "var(--color-text-secondary)" }}>Loading findings…</p>
        ) : null}

        {emptyVariant ? (
          <FindingEmptyState
            variant={emptyVariant}
            onScan={handleScan}
            onClearFilters={() => setFilters(defaultFindingFilters())}
          />
        ) : (
          <FindingsList groups={groups} onOpen={(id) => void openDetailTab(id)} />
        )}
      </div>

      {detailTabs.map((tab) => {
        const record = findings.find((item) => item.id === tab.findingId);
        const key = tabKeyOf(tab);
        const detail = record
          ? mapFindingDetail(record, lifecycleById[record.id] ?? [])
          : null;
        return (
          <div
            key={key}
            role="tabpanel"
            id={tabPanelId(key)}
            aria-labelledby={tabId(key)}
            hidden={activeTabKey !== key}
          >
            {detail ? (
              <FindingDetail
                detail={detail}
                busy={actionBusy}
                onAcknowledge={() =>
                  void runLifecycle(
                    () => gateway.acknowledge(detail.finding.id),
                    "Finding acknowledged.",
                  )
                }
                onDismiss={(reason: FindingsDismissReason) =>
                  void runLifecycle(
                    () => gateway.dismiss(detail.finding.id, reason),
                    "Finding dismissed.",
                  )
                }
                onSuppress={(preset: FindingsSuppressPreset) =>
                  void runLifecycle(
                    () => gateway.suppress(detail.finding.id, preset),
                    "Finding suppressed.",
                  )
                }
                onOpenEnvironment={() =>
                  setBanner(
                    "Open Environments from the sidebar to review this environment.",
                  )
                }
                onOpenIntegration={() =>
                  setBanner(
                    "Open Integrations from the sidebar to review this connection.",
                  )
                }
                onResync={() =>
                  setBanner("Sync started (stub) — use Environments Sync for full sync.")
                }
              />
            ) : (
              <p style={{ color: "var(--color-text-secondary)" }}>
                Finding not found.
              </p>
            )}
          </div>
        );
      })}
    </section>
  );
}
