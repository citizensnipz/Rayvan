import { AppShell, Button, EmptyState } from "@rayvan/ui";

export function App() {
  return (
    <AppShell>
      <header>
        <h1 style={{ margin: 0, fontSize: "2rem" }}>Rayvan</h1>
        <p style={{ margin: "0.5rem 0 0", color: "#475569" }}>
          Local-first infrastructure control plane.
        </p>
      </header>

      <EmptyState
        title="No projects connected yet."
        description="Connect a local project to begin discovering infrastructure."
      />

      <div style={{ marginTop: "1rem" }}>
        <Button disabled>Add project</Button>
      </div>
    </AppShell>
  );
}
