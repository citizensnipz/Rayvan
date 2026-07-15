import { useMemo, useState, type CSSProperties, type KeyboardEvent } from "react";
import {
  filterConfigurationMatrix,
  type ConfigurationMatrixCellStatus,
  type ConfigurationMatrixViewModel,
} from "@rayvan/config-engine";
import { Input } from "@rayvan/ui";

import type { ConfigurationCellSelection } from "./view-models.js";

const STATUS_FILTERS: Array<{ id: ConfigurationMatrixCellStatus; label: string }> = [
  { id: "healthy", label: "Healthy" },
  { id: "missing", label: "Missing" },
  { id: "mismatched", label: "Mismatched" },
  { id: "locked", label: "Locked" },
  { id: "present", label: "Present" },
];

const scrollStyle: CSSProperties = {
  overflowX: "auto",
  border: "1px solid var(--color-border)",
  borderRadius: "8px",
  background: "var(--color-surface)",
};

const stickyKeyStyle: CSSProperties = {
  position: "sticky",
  left: 0,
  background: "var(--color-surface)",
  zIndex: 2,
  minWidth: "12rem",
  maxWidth: "16rem",
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
};

function cellBackground(status: ConfigurationMatrixCellStatus): string {
  switch (status) {
    case "healthy":
      return "color-mix(in srgb, #16a34a 12%, var(--color-surface))";
    case "missing":
      return "color-mix(in srgb, #dc2626 12%, var(--color-surface))";
    case "mismatched":
      return "color-mix(in srgb, #d97706 14%, var(--color-surface))";
    case "locked":
      return "color-mix(in srgb, #64748b 14%, var(--color-surface))";
    default:
      return "var(--color-surface)";
  }
}

interface ConfigurationMatrixProps {
  matrix: ConfigurationMatrixViewModel | null;
  onSelectCell: (selection: ConfigurationCellSelection) => void;
  onOpenKey: (configurationKeyId: string, label: string) => void;
  onOpenEnvironment: (environmentId: string, label: string) => void;
}

export function ConfigurationMatrix({
  matrix,
  onSelectCell,
  onOpenKey,
  onOpenEnvironment,
}: ConfigurationMatrixProps) {
  const [search, setSearch] = useState("");
  const [requiredOnly, setRequiredOnly] = useState(false);
  const [sensitiveOnly, setSensitiveOnly] = useState(false);
  const [statuses, setStatuses] = useState<ConfigurationMatrixCellStatus[]>([]);

  const filtered = useMemo(() => {
    if (!matrix) {
      return null;
    }
    return filterConfigurationMatrix(matrix, {
      search,
      requiredOnly,
      sensitiveOnly,
      statuses: statuses.length > 0 ? statuses : undefined,
    });
  }, [matrix, search, requiredOnly, sensitiveOnly, statuses]);

  if (!matrix || !filtered) {
    return (
      <p style={{ color: "var(--color-text-secondary)" }}>
        Configuration matrix will appear after environments and keys are available.
      </p>
    );
  }

  function toggleStatus(status: ConfigurationMatrixCellStatus) {
    setStatuses((current) =>
      current.includes(status)
        ? current.filter((item) => item !== status)
        : [...current, status],
    );
  }

  function handleCellKeyDown(
    event: KeyboardEvent<HTMLButtonElement>,
    rowIndex: number,
    cellIndex: number,
  ) {
    const rowCount = filtered!.rows.length;
    const colCount = filtered!.columns.length;
    let nextRow = rowIndex;
    let nextCol = cellIndex;
    if (event.key === "ArrowRight") {
      nextCol = Math.min(colCount - 1, cellIndex + 1);
    } else if (event.key === "ArrowLeft") {
      nextCol = Math.max(0, cellIndex - 1);
    } else if (event.key === "ArrowDown") {
      nextRow = Math.min(rowCount - 1, rowIndex + 1);
    } else if (event.key === "ArrowUp") {
      nextRow = Math.max(0, rowIndex - 1);
    } else {
      return;
    }
    event.preventDefault();
    const target = document.querySelector<HTMLElement>(
      `[data-matrix-cell="${nextRow}:${nextCol}"]`,
    );
    target?.focus();
  }

  return (
    <section>
      <h2 style={{ marginTop: 0 }}>Configuration Matrix</h2>
      <p style={{ color: "var(--color-text-secondary)" }}>
        Compare keys across environments. Secret values stay masked or locked.
      </p>

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "0.75rem",
          marginBottom: "1rem",
          alignItems: "center",
        }}
      >
        <Input
          aria-label="Search configuration keys"
          placeholder="Search keys…"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          style={{ maxWidth: "16rem" }}
        />
        <label style={{ display: "inline-flex", gap: "0.35rem", alignItems: "center" }}>
          <input
            type="checkbox"
            checked={requiredOnly}
            onChange={(event) => setRequiredOnly(event.target.checked)}
          />
          Required only
        </label>
        <label style={{ display: "inline-flex", gap: "0.35rem", alignItems: "center" }}>
          <input
            type="checkbox"
            checked={sensitiveOnly}
            onChange={(event) => setSensitiveOnly(event.target.checked)}
          />
          Sensitive only
        </label>
        {STATUS_FILTERS.map((filter) => (
          <label
            key={filter.id}
            style={{ display: "inline-flex", gap: "0.35rem", alignItems: "center" }}
          >
            <input
              type="checkbox"
              checked={statuses.includes(filter.id)}
              onChange={() => toggleStatus(filter.id)}
            />
            {filter.label}
          </label>
        ))}
      </div>

      <div style={scrollStyle}>
        <table
          role="grid"
          aria-label="Configuration matrix"
          style={{
            borderCollapse: "collapse",
            minWidth: "100%",
            fontSize: "0.85rem",
          }}
        >
          <thead>
            <tr>
              <th scope="col" style={{ ...stickyKeyStyle, padding: "0.65rem", textAlign: "left" }}>
                Key
              </th>
              {filtered.columns.map((column) => (
                <th
                  key={column.environmentId}
                  scope="col"
                  style={{
                    padding: "0.65rem",
                    textAlign: "left",
                    minWidth: "8rem",
                    maxWidth: "12rem",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  <button
                    type="button"
                    onClick={() => onOpenEnvironment(column.environmentId, column.name)}
                    style={{
                      border: "none",
                      background: "transparent",
                      color: "var(--color-text)",
                      cursor: "pointer",
                      fontWeight: 600,
                      padding: 0,
                    }}
                    title={column.name}
                  >
                    {column.name}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.rows.map((row, rowIndex) => (
              <tr key={row.configurationKeyId}>
                <th
                  scope="row"
                  style={{ ...stickyKeyStyle, padding: "0.5rem 0.65rem", fontWeight: 600 }}
                >
                  <button
                    type="button"
                    onClick={() => onOpenKey(row.configurationKeyId, row.name)}
                    style={{
                      border: "none",
                      background: "transparent",
                      color: "var(--color-text)",
                      cursor: "pointer",
                      fontWeight: 600,
                      padding: 0,
                      maxWidth: "100%",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                    title={row.name}
                  >
                    {row.name}
                  </button>
                  <div style={{ fontSize: "0.75rem", color: "var(--color-text-muted)" }}>
                    {row.required ? "Required" : "Optional"}
                    {row.sensitive ? " · Sensitive" : ""}
                  </div>
                </th>
                {row.cells.map((cell, cellIndex) => {
                  const column = filtered.columns[cellIndex];
                  const display =
                    cell.accessLocked
                      ? "Locked"
                      : cell.safeVisibleValue ?? cell.statusLabel;
                  return (
                    <td key={`${cell.configurationKeyId}:${cell.environmentId}`} style={{ padding: 0 }}>
                      <button
                        type="button"
                        role="gridcell"
                        data-matrix-cell={`${rowIndex}:${cellIndex}`}
                        aria-label={`${row.name} in ${column?.name ?? "environment"}: ${cell.statusLabel}`}
                        title={cell.statusLabel}
                        onClick={() =>
                          onSelectCell({
                            configurationKeyId: cell.configurationKeyId,
                            configurationKeyName: cell.configurationKeyName,
                            environmentId: cell.environmentId,
                            environmentName: column?.name ?? "Environment",
                            status: cell.status,
                            statusLabel: cell.statusLabel,
                            safeVisibleValue: cell.safeVisibleValue,
                            accessLocked: cell.accessLocked,
                            requiredMissing: cell.requiredMissing,
                            occurrenceIds: cell.occurrenceIds,
                          })
                        }
                        onKeyDown={(event) => handleCellKeyDown(event, rowIndex, cellIndex)}
                        style={{
                          width: "100%",
                          minHeight: "3rem",
                          border: "1px solid var(--color-border)",
                          background: cellBackground(cell.status),
                          color: "var(--color-text)",
                          cursor: "pointer",
                          textAlign: "left",
                          padding: "0.5rem 0.65rem",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        }}
                      >
                        <span aria-hidden="true">{display}</span>
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
