import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { IntegrationCard } from "./IntegrationCard.js";
import { IntegrationIcon } from "./icons.js";
import { resolveIntegrationTheme, sanitizeAccentColor } from "./theme.js";
import type { PluginIntegrationCardViewModel } from "./view-models.js";

function buildCard(
  overrides: Partial<PluginIntegrationCardViewModel> = {},
): PluginIntegrationCardViewModel {
  return {
    connectionId: "connection-1",
    installedPluginId: "installed-1",
    pluginId: "vercel",
    connectionName: "Acme Organisation",
    pluginName: "Vercel",
    publisher: "rayvan",
    icon: { iconId: "vercel", initials: "V", label: "Vercel" },
    theme: { surface: "dark", accentColor: "#FFFFFF", foregroundMode: "light" },
    status: "connected",
    statusLabel: "Connected",
    fields: [{ label: "Project", value: "rayvan-web" }],
    actions: [
      { id: "open", label: "Open", kind: "primary" },
      { id: "sync", label: "Sync", kind: "secondary" },
      { id: "configure", label: "Configure", kind: "secondary" },
    ],
    ...overrides,
  };
}

describe("IntegrationCard", () => {
  it("renders name, status, and fields", () => {
    render(
      <IntegrationCard card={buildCard()} onOpen={vi.fn()} onAction={vi.fn()} />,
    );

    expect(screen.getByText("Acme Organisation")).toBeInTheDocument();
    expect(screen.getByText("Connected")).toBeInTheDocument();
    expect(screen.getByText("Project")).toBeInTheDocument();
    expect(screen.getByText("rayvan-web")).toBeInTheDocument();
  });

  it("calls onAction(open) when Open is clicked", () => {
    const onAction = vi.fn();
    render(
      <IntegrationCard card={buildCard()} onOpen={vi.fn()} onAction={onAction} />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Open" }));
    expect(onAction).toHaveBeenCalledWith("connection-1", "open");
  });

  it("calls onAction for secondary actions", () => {
    const onOpen = vi.fn();
    const onAction = vi.fn();
    render(<IntegrationCard card={buildCard()} onOpen={onOpen} onAction={onAction} />);

    fireEvent.click(screen.getByRole("button", { name: "Sync" }));
    expect(onAction).toHaveBeenCalledWith("connection-1", "sync");
    expect(onOpen).not.toHaveBeenCalled();
  });

  it("keeps Open and Sync as sibling buttons without nesting interactives", () => {
    render(
      <IntegrationCard card={buildCard()} onOpen={vi.fn()} onAction={vi.fn()} />,
    );
    expect(screen.getByRole("article", { name: "Vercel: Acme Organisation" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sync" })).toBeInTheDocument();
  });

  it("renders distinct indicators for each status, not color alone", () => {
    const { rerender } = render(
      <IntegrationCard
        card={buildCard({ status: "attention_required", statusLabel: "Attention required" })}
        onOpen={vi.fn()}
        onAction={vi.fn()}
      />,
    );
    expect(screen.getByText("Attention required")).toBeInTheDocument();

    rerender(
      <IntegrationCard
        card={buildCard({ status: "error", statusLabel: "Error" })}
        onOpen={vi.fn()}
        onAction={vi.fn()}
      />,
    );
    expect(screen.getByText("Error")).toBeInTheDocument();
  });
});

describe("IntegrationIcon fallback", () => {
  it("renders initials derived from the label when icon initials are missing", () => {
    const theme = resolveIntegrationTheme(undefined);
    render(<IntegrationIcon icon={{ label: "Custom Plugin" }} theme={theme} />);
    expect(screen.getByRole("img", { name: "Custom Plugin" })).toHaveTextContent("CU");
  });

  it("renders a generic fallback when no icon is provided at all", () => {
    const theme = resolveIntegrationTheme(undefined);
    render(<IntegrationIcon icon={undefined} theme={theme} />);
    expect(screen.getByRole("img", { name: "Integration" })).toBeInTheDocument();
  });
});

describe("theme sanitization", () => {
  it("rejects free-form accent CSS and keeps dark chips readable", () => {
    expect(sanitizeAccentColor("rgb(255,0,0)")).toBeUndefined();
    expect(sanitizeAccentColor("#FFFFFF")).toBe("#FFFFFF");

    const resolved = resolveIntegrationTheme({
      surface: "dark",
      accentColor: "#FFFFFF",
      foregroundMode: "light",
    });
    expect(resolved.iconBackground).toBe("#111827");
    expect(resolved.iconForeground).toBe("#f8fafc");
    expect(resolved.accentColor).toBe("#FFFFFF");
  });
});
