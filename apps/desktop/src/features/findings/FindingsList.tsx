import { FindingListItem } from "./FindingListItem.js";
import { SeverityBadge } from "./severity.js";
import type { FindingSeverityGroupViewModel } from "./view-models.js";

interface FindingsListProps {
  groups: FindingSeverityGroupViewModel[];
  selectedFindingId?: string | null;
  onOpen: (findingId: string) => void;
}

export function FindingsList({
  groups,
  selectedFindingId,
  onOpen,
}: FindingsListProps) {
  return (
    <div style={{ display: "grid", gap: "1.25rem" }}>
      {groups.map((group) => (
        <section
          key={group.severity}
          aria-label={`${group.severityLabel} findings`}
        >
          <h3
            style={{
              margin: "0 0 0.5rem",
              fontSize: "0.95rem",
              fontWeight: 600,
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
            }}
          >
            <SeverityBadge
              severity={group.severity}
              label={group.severityLabel}
              size="md"
            />
            <span
              style={{
                fontWeight: 400,
                color: "var(--color-text-secondary)",
                fontSize: "0.85rem",
              }}
            >
              ({group.items.length})
            </span>
          </h3>
          <ul
            style={{
              listStyle: "none",
              padding: 0,
              margin: 0,
              display: "grid",
              gap: "0.5rem",
            }}
          >
            {group.items.map((item) => (
              <li key={item.findingId}>
                <FindingListItem
                  item={item}
                  selected={selectedFindingId === item.findingId}
                  onOpen={onOpen}
                />
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}
