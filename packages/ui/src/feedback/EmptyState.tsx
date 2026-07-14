export function EmptyState({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <div
      style={{
        marginTop: "1.5rem",
        padding: "1.25rem",
        borderRadius: "8px",
        border: "1px dashed var(--color-border-strong)",
        background: "var(--color-surface)",
      }}
    >
      <strong>{title}</strong>
      <p style={{ margin: "0.5rem 0 0", color: "var(--color-text-secondary)" }}>
        {description}
      </p>
    </div>
  );
}
