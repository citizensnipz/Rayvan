import { useMemo, useState } from "react";
import type { Environment } from "@rayvan/core";
import { Button } from "@rayvan/ui";

import type { EnvironmentResourcesViewModel, ResourceListItemViewModel } from "./view-models.js";

interface EnvironmentResourcesProps {
  viewModel: EnvironmentResourcesViewModel;
  environments: Environment[];
  onAttach: (discoveredResourceId: string, environmentId: string) => void;
  onMove: (bindingId: string, environmentId: string) => void;
  onDetach: (bindingId: string) => void;
}

export function EnvironmentResources({
  viewModel,
  environments,
  onAttach,
  onMove,
  onDetach,
}: EnvironmentResourcesProps) {
  const [pendingAttach, setPendingAttach] = useState<Record<string, string>>({});
  const [pendingMove, setPendingMove] = useState<Record<string, string>>({});

  const environmentOptions = useMemo(
    () =>
      environments.map((environment) => ({
        id: environment.id,
        name: environment.name,
      })),
    [environments],
  );

  function renderActions(item: ResourceListItemViewModel) {
    if (item.bindingId && item.environmentId) {
      const moveTarget =
        pendingMove[item.bindingId] ??
        environmentOptions.find((option) => option.id !== item.environmentId)?.id ??
        "";
      return (
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
          <select
            aria-label={`Move ${item.name} to environment`}
            value={moveTarget}
            onChange={(event) =>
              setPendingMove((current) => ({
                ...current,
                [item.bindingId!]: event.target.value,
              }))
            }
            style={{
              padding: "0.35rem 0.5rem",
              borderRadius: "6px",
              border: "1px solid var(--color-border-strong)",
              background: "var(--color-surface)",
            }}
          >
            {environmentOptions
              .filter((option) => option.id !== item.environmentId)
              .map((option) => (
                <option key={option.id} value={option.id}>
                  {option.name}
                </option>
              ))}
          </select>
          <Button
            disabled={!moveTarget}
            onClick={() => onMove(item.bindingId!, moveTarget)}
          >
            Move
          </Button>
          <Button onClick={() => onDetach(item.bindingId!)}>Detach</Button>
        </div>
      );
    }

    const attachTarget =
      pendingAttach[item.discoveredResourceId] ?? environmentOptions[0]?.id ?? "";
    return (
      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
        <select
          aria-label={`Attach ${item.name} to environment`}
          value={attachTarget}
          onChange={(event) =>
            setPendingAttach((current) => ({
              ...current,
              [item.discoveredResourceId]: event.target.value,
            }))
          }
          style={{
            padding: "0.35rem 0.5rem",
            borderRadius: "6px",
            border: "1px solid var(--color-border-strong)",
            background: "var(--color-surface)",
          }}
        >
          {environmentOptions.map((option) => (
            <option key={option.id} value={option.id}>
              {option.name}
            </option>
          ))}
        </select>
        <Button
          disabled={!attachTarget}
          onClick={() => onAttach(item.discoveredResourceId, attachTarget)}
        >
          Attach
        </Button>
      </div>
    );
  }

  return (
    <section>
      <h2 style={{ marginTop: 0 }}>Resources</h2>
      <p style={{ color: "var(--color-text-secondary)" }}>
        Bound resources are grouped by environment. Detach never deletes the discovered resource.
      </p>

      {viewModel.groups.map((group) => (
        <div key={group.title} style={{ marginBottom: "1.5rem" }}>
          <h3 style={{ marginBottom: "0.65rem" }}>{group.title}</h3>
          {group.items.length === 0 ? (
            <p style={{ color: "var(--color-text-muted)" }}>No resources in this group.</p>
          ) : (
            <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: "0.55rem" }}>
              {group.items.map((item) => (
                <li
                  key={item.discoveredResourceId}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    gap: "1rem",
                    flexWrap: "wrap",
                    padding: "0.75rem",
                    border: "1px solid var(--color-border)",
                    borderRadius: "8px",
                    background: "var(--color-surface)",
                  }}
                >
                  <div style={{ minWidth: 0 }}>
                    <strong
                      title={item.name}
                      style={{
                        display: "block",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {item.name}
                    </strong>
                    <div style={{ fontSize: "0.85rem", color: "var(--color-text-secondary)" }}>
                      {item.pluginId} · {item.resourceType}
                    </div>
                  </div>
                  {renderActions(item)}
                </li>
              ))}
            </ul>
          )}
        </div>
      ))}
    </section>
  );
}
