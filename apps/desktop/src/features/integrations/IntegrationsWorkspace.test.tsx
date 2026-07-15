import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { createDevPluginIntegrationsGateway } from "../../lib/plugins/index.js";
import type { PluginIntegrationsGateway } from "../../lib/plugins/index.js";
import { IntegrationsProvider } from "./IntegrationsContext.js";
import { IntegrationsWorkspace } from "./IntegrationsWorkspace.js";

/** A dev gateway whose catalog is installed, but which never seeds project connections. */
function createEmptyGateway(): PluginIntegrationsGateway {
  const real = createDevPluginIntegrationsGateway();
  return { ...real, ensureProjectSeeded: async () => {} };
}

function renderWorkspace(gateway: PluginIntegrationsGateway, projectId: string | null) {
  return render(
    <IntegrationsProvider gateway={gateway}>
      <IntegrationsWorkspace projectId={projectId} />
    </IntegrationsProvider>,
  );
}

function openCard(name: string) {
  const article = screen.getByRole("article", { name: new RegExp(name, "i") });
  fireEvent.click(within(article).getByRole("button", { name: "Open" }));
}

describe("IntegrationsWorkspace", () => {
  it("shows the empty state when the project has no connections", async () => {
    renderWorkspace(createEmptyGateway(), "project-1");

    expect(await screen.findByText("No integrations configured")).toBeInTheDocument();
    expect(
      screen.getAllByText("Connect Rayvan to the services used by this project."),
    ).not.toHaveLength(0);
  });

  it("only renders cards for the current project's connections", async () => {
    renderWorkspace(createDevPluginIntegrationsGateway(), "project-1");

    await screen.findByText("Acme Organisation");
    expect(screen.getAllByRole("article")).toHaveLength(5);
  });

  it("renders the seeded Sentry connection with an attention-required status", async () => {
    renderWorkspace(createDevPluginIntegrationsGateway(), "project-1");

    await screen.findByText("Rayvan Production");
    expect(screen.getByText("Attention required")).toBeInTheDocument();
  });

  it("opens a tab when Open is clicked and reuses the existing tab on a second click", async () => {
    renderWorkspace(createDevPluginIntegrationsGateway(), "project-1");
    await screen.findByText("Acme Organisation");

    openCard("Acme Organisation");
    expect(await screen.findByRole("tab", { name: "Acme Organisation" })).toBeInTheDocument();
    expect(screen.getAllByRole("tab")).toHaveLength(2);

    fireEvent.click(screen.getByRole("tab", { name: "Home" }));
    openCard("Acme Organisation");

    expect(screen.getAllByRole("tab")).toHaveLength(2);
    expect(screen.getByRole("tab", { name: "Acme Organisation" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  it("does not allow closing the Home tab, but allows closing detail tabs", async () => {
    renderWorkspace(createDevPluginIntegrationsGateway(), "project-1");
    await screen.findByText("Acme Organisation");

    openCard("Acme Organisation");
    await screen.findByRole("tab", { name: "Acme Organisation" });

    expect(screen.queryByRole("button", { name: "Close Home tab" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Close Acme Organisation tab" }));

    expect(screen.queryByRole("tab", { name: "Acme Organisation" })).not.toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Home" })).toHaveAttribute("aria-selected", "true");
  });

  it("resets tabs when the current project changes", async () => {
    const gateway = createDevPluginIntegrationsGateway();
    const { rerender } = renderWorkspace(gateway, "project-1");
    await screen.findByText("Acme Organisation");

    openCard("Acme Organisation");
    await screen.findByRole("tab", { name: "Acme Organisation" });

    rerender(
      <IntegrationsProvider gateway={gateway}>
        <IntegrationsWorkspace projectId="project-2" />
      </IntegrationsProvider>,
    );

    await waitFor(() => {
      expect(screen.getAllByRole("tab")).toHaveLength(1);
    });
    expect(screen.getByRole("tab", { name: "Home" })).toHaveAttribute("aria-selected", "true");
  });

  it("uses accessible tab roles, panels, and labels", async () => {
    renderWorkspace(createDevPluginIntegrationsGateway(), "project-1");
    await screen.findByText("Acme Organisation");

    expect(screen.getByRole("tablist", { name: "Integrations tabs" })).toBeInTheDocument();
    const homeTab = screen.getByRole("tab", { name: "Home" });
    expect(homeTab).toHaveAttribute("aria-selected", "true");
    expect(homeTab).toHaveAttribute("aria-controls", "integrations-panel-home");

    openCard("Acme Organisation");
    const detailTab = await screen.findByRole("tab", { name: "Acme Organisation" });
    expect(detailTab).toHaveAttribute("aria-selected", "true");
    expect(homeTab).toHaveAttribute("aria-selected", "false");
    expect(screen.getByRole("tabpanel", { name: "Acme Organisation" })).toBeInTheDocument();
  });

  it("opens the Add integration dialog", async () => {
    renderWorkspace(createEmptyGateway(), "project-1");
    await screen.findByText("No integrations configured");

    fireEvent.click(screen.getAllByRole("button", { name: /Add integration/ })[0]!);
    expect(screen.getByRole("dialog", { name: "Add integration" })).toBeInTheDocument();
  });

  it("closes the Add integration dialog with Escape", async () => {
    renderWorkspace(createEmptyGateway(), "project-1");
    await screen.findByText("No integrations configured");

    fireEvent.click(screen.getAllByRole("button", { name: /Add integration/ })[0]!);
    expect(screen.getByRole("dialog", { name: "Add integration" })).toBeInTheDocument();

    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() => {
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });
  });

  it("lists eligible catalog plugins in the library, honoring supportsMultipleConnections", async () => {
    renderWorkspace(createDevPluginIntegrationsGateway(), "project-1");
    await screen.findByText("Acme Organisation");

    fireEvent.click(screen.getAllByRole("button", { name: /\+ Add integration/ })[0]!);
    fireEvent.click(screen.getByRole("button", { name: "Add from library" }));
    const dialog = screen.getByRole("dialog", { name: "Add integration" });

    // Vercel already has a connection but supports multiple, so it stays addable.
    expect(await within(dialog).findByText("Vercel")).toBeInTheDocument();
    // RunPod has no connection yet.
    expect(within(dialog).getByText("RunPod")).toBeInTheDocument();
    // Example Local supports only a single connection and is already configured.
    expect(within(dialog).queryByText("Example Local")).not.toBeInTheDocument();
  });

  it("creates a connection from the library, shows it as a card, and opens its detail tab", async () => {
    renderWorkspace(createDevPluginIntegrationsGateway(), "project-1");
    await screen.findByText("Acme Organisation");

    fireEvent.click(screen.getAllByRole("button", { name: /\+ Add integration/ })[0]!);
    fireEvent.click(screen.getByRole("button", { name: "Add from library" }));
    const dialog = screen.getByRole("dialog", { name: "Add integration" });
    await within(dialog).findByText("RunPod");

    fireEvent.click(within(dialog).getByRole("button", { name: "Configure" }));
    fireEvent.click(within(dialog).getByRole("button", { name: "Add integration" }));

    await waitFor(() => {
      expect(screen.getByText(/RunPod was connected successfully\./)).toBeInTheDocument();
    });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "RunPod" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Home" }));
    expect(screen.getAllByRole("article")).toHaveLength(6);
  });
});
