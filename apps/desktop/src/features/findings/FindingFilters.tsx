import type {
  FindingCategory,
  FindingSeverity,
  FindingStatus,
} from "@rayvan/core";

import {
  CATEGORY_LABELS,
  SEVERITY_LABELS,
  STATUS_LABELS,
  type FindingEnvironmentOption,
  type FindingFiltersState,
  type FindingIntegrationOption,
} from "./view-models.js";

const fieldStyle = {
  display: "flex",
  flexDirection: "column" as const,
  gap: "0.25rem",
  fontSize: "0.85rem",
};

const controlStyle = {
  padding: "0.4rem 0.55rem",
  borderRadius: "6px",
  border: "1px solid var(--color-border)",
  background: "var(--color-surface)",
  color: "var(--color-text)",
};

interface FindingFiltersProps {
  filters: FindingFiltersState;
  environments: FindingEnvironmentOption[];
  integrations: FindingIntegrationOption[];
  onChange: (next: FindingFiltersState) => void;
}

export function FindingFilters({
  filters,
  environments,
  integrations,
  onChange,
}: FindingFiltersProps) {
  return (
    <div
      role="search"
      aria-label="Filter findings"
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(11rem, 1fr))",
        gap: "0.75rem",
        marginBottom: "1rem",
        padding: "0.85rem",
        border: "1px solid var(--color-border)",
        borderRadius: "8px",
        background: "var(--color-surface-muted)",
      }}
    >
      <label style={{ ...fieldStyle, gridColumn: "1 / -1" }}>
        Search findings
        <input
          type="search"
          value={filters.search}
          placeholder="Title, summary, rule…"
          aria-label="Search findings"
          style={controlStyle}
          onChange={(event) =>
            onChange({ ...filters, search: event.target.value })
          }
        />
      </label>

      <label style={fieldStyle}>
        <span>
          <input
            type="checkbox"
            checked={filters.openOnly}
            aria-label="Show open findings only"
            onChange={(event) =>
              onChange({ ...filters, openOnly: event.target.checked })
            }
          />{" "}
          Show open findings only
        </span>
      </label>

      <label style={fieldStyle}>
        <span>
          <input
            type="checkbox"
            checked={filters.remediableOnly}
            aria-label="Remediable findings only"
            onChange={(event) =>
              onChange({ ...filters, remediableOnly: event.target.checked })
            }
          />{" "}
          Remediable only
        </span>
      </label>

      <label style={fieldStyle}>
        Status
        <select
          aria-label="Filter by status"
          disabled={filters.openOnly}
          value={filters.statuses[0] ?? ""}
          style={controlStyle}
          onChange={(event) => {
            const value = event.target.value as FindingStatus | "";
            onChange({
              ...filters,
              statuses: value ? [value] : [],
            });
          }}
        >
          <option value="">All statuses</option>
          {(Object.keys(STATUS_LABELS) as FindingStatus[]).map((status) => (
            <option key={status} value={status}>
              {STATUS_LABELS[status]}
            </option>
          ))}
        </select>
      </label>

      <label style={fieldStyle}>
        Severity
        <select
          aria-label="Filter by severity"
          value={filters.severities[0] ?? ""}
          style={controlStyle}
          onChange={(event) => {
            const value = event.target.value as FindingSeverity | "";
            onChange({
              ...filters,
              severities: value ? [value] : [],
            });
          }}
        >
          <option value="">All severities</option>
          {(Object.keys(SEVERITY_LABELS) as FindingSeverity[]).map((severity) => (
            <option key={severity} value={severity}>
              {SEVERITY_LABELS[severity]}
            </option>
          ))}
        </select>
      </label>

      <label style={fieldStyle}>
        Category
        <select
          aria-label="Filter by category"
          value={filters.categories[0] ?? ""}
          style={controlStyle}
          onChange={(event) => {
            const value = event.target.value as FindingCategory | "";
            onChange({
              ...filters,
              categories: value ? [value] : [],
            });
          }}
        >
          <option value="">All categories</option>
          {(Object.keys(CATEGORY_LABELS) as FindingCategory[]).map((category) => (
            <option key={category} value={category}>
              {CATEGORY_LABELS[category]}
            </option>
          ))}
        </select>
      </label>

      <label style={fieldStyle}>
        Environment
        <select
          aria-label="Filter by environment"
          value={filters.environmentId ?? ""}
          style={controlStyle}
          onChange={(event) =>
            onChange({
              ...filters,
              environmentId: event.target.value || null,
            })
          }
        >
          <option value="">All environments</option>
          {environments.map((environment) => (
            <option key={environment.id} value={environment.id}>
              {environment.label}
            </option>
          ))}
        </select>
      </label>

      <label style={fieldStyle}>
        Integration / plugin
        <select
          aria-label="Filter by integration"
          value={filters.connectionId ?? ""}
          style={controlStyle}
          onChange={(event) =>
            onChange({
              ...filters,
              connectionId: event.target.value || null,
            })
          }
        >
          <option value="">All integrations</option>
          {integrations.map((integration) => (
            <option key={integration.id} value={integration.id}>
              {integration.label}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
