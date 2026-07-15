import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import {
  createDevEnvironmentsGateway,
  type EnvironmentsGateway,
} from "../../lib/environments/index.js";
import { EnvironmentsProvider } from "./EnvironmentsContext.js";
import { EnvironmentsWorkspace } from "./EnvironmentsWorkspace.js";

afterEach(() => {
  cleanup();
});

function renderWorkspace(gateway: EnvironmentsGateway, projectId: string) {
  return render(
    <EnvironmentsProvider gateway={gateway}>
      <EnvironmentsWorkspace projectId={projectId} />
    </EnvironmentsProvider>,
  );
}

async function openEnvironmentConfiguration(
  environmentName: RegExp,
  projectId: string,
) {
  const gateway = createDevEnvironmentsGateway();
  renderWorkspace(gateway, projectId);

  const card = await screen.findByRole("article", { name: environmentName });
  fireEvent.click(within(card).getByRole("button", { name: "Open" }));
  fireEvent.click(await screen.findByRole("tab", { name: "Configuration" }));
  return gateway;
}

describe("Environment configuration editor", () => {
  it("lists keys, edits values, and distinguishes unsaved vs not applied", async () => {
    await openEnvironmentConfiguration(/Production/i, "project-config-editor");

    const apiInput = await screen.findByLabelText("Desired value for API_BASE_URL");
    expect(apiInput).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save locally" })).toBeDisabled();

    fireEvent.change(apiInput, { target: { value: "https://api.edited.example.com" } });

    expect(await screen.findByText("Unsaved local changes")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save locally" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Apply to integrations" })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Discard" }));
    await waitFor(() => {
      expect(screen.queryByText("Unsaved local changes")).not.toBeInTheDocument();
    });
  });

  it("hides sensitive values by default and does not reveal locked remotes", async () => {
    await openEnvironmentConfiguration(/Production/i, "project-config-secrets");

    expect(await screen.findByLabelText(/Desired value for GITHUB_TOKEN/i)).toBeInTheDocument();

    // Sensitive desired fields should not dump plaintext fixture secrets into the document.
    expect(screen.queryByText(/sk_live_fake/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/postgres:\/\//i)).not.toBeInTheDocument();

    const showButtons = screen.getAllByRole("button", { name: /Show value for/i });
    expect(showButtons.length).toBeGreaterThan(0);
  });

  it("does not mark sensitive fields dirty on focus alone", async () => {
    await openEnvironmentConfiguration(/Development/i, "project-config-sensitive-dirty");

    const dbInput = await screen.findByLabelText(/Desired value for DATABASE_URL/i);
    expect(screen.getByRole("button", { name: "Save locally" })).toBeDisabled();

    fireEvent.focus(dbInput);
    expect(screen.getByRole("button", { name: "Save locally" })).toBeDisabled();
    expect(screen.queryByText("Unsaved local changes")).not.toBeInTheDocument();

    fireEvent.change(dbInput, { target: { value: "fake-new-secret-value" } });
    expect(await screen.findByText("Unsaved local changes")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save locally" })).toBeEnabled();
  });

  it("builds a redacted apply plan without provider calls and preserves desired on partial apply", async () => {
    const gateway = await openEnvironmentConfiguration(
      /Staging/i,
      "project-config-apply",
    );

    const applyButton = await screen.findByRole("button", {
      name: "Apply to integrations",
    });
    fireEvent.click(applyButton);

    const dialog = await screen.findByRole("dialog", {
      name: "Review configuration apply plan",
    });
    expect(within(dialog).getByText(/no provider API calls/i)).toBeInTheDocument();
    expect(screen.queryByText(/sk_live|postgres:\/\/[^•]/i)).not.toBeInTheDocument();

    fireEvent.click(within(dialog).getByRole("button", { name: "Approve and apply" }));
    expect(await within(dialog).findByText(/Apply result:/i)).toBeInTheDocument();

    const staging = (await gateway.listEnvironments("project-config-apply")).find(
      (environment) => environment.name === "Staging",
    )!;
    const desired = await gateway.listDesiredValuesByEnvironment(staging.id);
    expect(desired.length).toBeGreaterThan(0);
  });

  it("shows Compare with other environments as a secondary matrix action", async () => {
    await openEnvironmentConfiguration(/Staging/i, "project-config-compare");

    const compareButtons = await screen.findAllByRole("button", {
      name: "Compare with other environments",
    });
    fireEvent.click(compareButtons[0]!);
    expect(
      await screen.findByRole("tab", { name: "Configuration Matrix", selected: true }),
    ).toBeInTheDocument();
  });
});
