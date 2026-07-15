import { Button, EmptyState } from "@rayvan/ui";

interface EnvironmentEmptyStateProps {
  onCreateEnvironment?: () => void;
  onOpenMatrix?: () => void;
  variant?: "no-environments" | "no-project";
}

export function EnvironmentEmptyState({
  onCreateEnvironment,
  onOpenMatrix,
  variant = "no-environments",
}: EnvironmentEmptyStateProps) {
  if (variant === "no-project") {
    return (
      <EmptyState
        title="Select a project"
        description="Choose a project to manage environments, configuration, and resource mappings."
      />
    );
  }

  return (
    <div>
      <EmptyState
        title="No environments yet"
        description="Create a local environment to start comparing configuration across Development, Staging, Production, and more."
      />
      <div style={{ display: "flex", gap: "0.5rem", marginTop: "1rem", flexWrap: "wrap" }}>
        {onCreateEnvironment ? (
          <Button onClick={onCreateEnvironment}>Create environment</Button>
        ) : null}
        {onOpenMatrix ? (
          <Button onClick={onOpenMatrix}>Open Configuration Matrix</Button>
        ) : null}
      </div>
    </div>
  );
}
