import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  createDevFindingsGateway,
  type FindingsGateway,
} from "../../lib/findings/index.js";
import { FindingsProvider } from "./FindingsContext.js";
import { FindingsWorkspace } from "./FindingsWorkspace.js";

function createEmptyGateway(): FindingsGateway {
  const real = createDevFindingsGateway();
  return {
    ...real,
    ensureProjectSeeded: async () => {},
    listFindings: async () => [],
    getProjectSummary: async () => ({
      openCount: 0,
      acknowledgedCount: 0,
      bySeverity: { info: 0, warning: 0, error: 0, critical: 0 },
      byCategory: {},
      hasRemediableFindings: false,
    }),
    getEvaluationState: async (projectId) => ({
      projectId,
      inProgress: false,
      phase: "idle",
      cancelled: false,
    }),
  };
}

function renderWorkspace(gateway: FindingsGateway, projectId: string | null) {
  return render(
    <FindingsProvider gateway={gateway}>
      <FindingsWorkspace projectId={projectId} />
    </FindingsProvider>,
  );
}

describe("FindingsWorkspace", () => {
  it("shows empty state when there are no findings", async () => {
    renderWorkspace(createEmptyGateway(), "project-1");

    expect(await screen.findByText("No findings yet")).toBeInTheDocument();
  });

  it("defaults to open findings only and groups by severity", async () => {
    renderWorkspace(createDevFindingsGateway(), "project-1");

    expect(await screen.findByRole("heading", { name: "Findings" })).toBeInTheDocument();
    expect(screen.getByLabelText("Show open findings only")).toBeChecked();

    expect(
      await screen.findByRole("region", { name: /Critical findings/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("region", { name: /Warning findings/i })).toBeInTheDocument();

    // Dismissed fixture should be hidden by default open-only filter
    expect(
      screen.queryByRole("button", { name: /Unmanaged key: VERCEL_URL/i }),
    ).not.toBeInTheDocument();
  });

  it("filters by search", async () => {
    renderWorkspace(createDevFindingsGateway(), "project-1");
    await screen.findByRole("region", { name: /Critical findings/i });

    fireEvent.change(screen.getByLabelText("Search findings"), {
      target: { value: "STRIPE_SECRET_KEY" },
    });

    expect(
      await screen.findByRole("button", {
        name: /Missing required configuration: STRIPE_SECRET_KEY/i,
      }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /Configuration mismatch: API_BASE_URL/i }),
    ).not.toBeInTheDocument();
  });

  it("opens finding detail", async () => {
    renderWorkspace(createDevFindingsGateway(), "project-1");
    const button = await screen.findByRole("button", {
      name: /Missing required configuration: STRIPE_SECRET_KEY/i,
    });
    fireEvent.click(button);

    expect(
      await screen.findByRole("tab", {
        name: /Missing required configuration: STRIPE_SECRET_KEY/i,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("article", {
        name: /Finding detail: Missing required configuration: STRIPE_SECRET_KEY/i,
      }),
    ).toBeInTheDocument();
  });

  it("acknowledges a finding", async () => {
    renderWorkspace(createDevFindingsGateway(), "project-1");
    const button = await screen.findByRole("button", {
      name: /Missing required configuration: STRIPE_SECRET_KEY/i,
    });
    fireEvent.click(button);

    const detail = await screen.findByRole("article", {
      name: /Finding detail: Missing required configuration: STRIPE_SECRET_KEY/i,
    });
    fireEvent.click(within(detail).getByRole("button", { name: "Acknowledge" }));

    expect(await screen.findByText("Finding acknowledged.")).toBeInTheDocument();
    await waitFor(() => {
      expect(within(detail).getByLabelText("Status: Acknowledged")).toBeInTheDocument();
    });
  });

  it("dismisses a finding with a reason", async () => {
    renderWorkspace(createDevFindingsGateway(), "project-1");
    const button = await screen.findByRole("button", {
      name: /Configuration mismatch: API_BASE_URL/i,
    });
    fireEvent.click(button);

    const detail = await screen.findByRole("article", {
      name: /Finding detail: Configuration mismatch: API_BASE_URL/i,
    });
    fireEvent.change(within(detail).getByLabelText("Dismissal reason"), {
      target: { value: "False positive" },
    });
    fireEvent.click(within(detail).getByRole("button", { name: "Dismiss" }));

    expect(await screen.findByText("Finding dismissed.")).toBeInTheDocument();
  });

  it("hides sensitive evidence values from accessible plaintext", async () => {
    renderWorkspace(createDevFindingsGateway(), "project-1");
    const button = await screen.findByRole("button", {
      name: /Locked comparison: DATABASE_URL/i,
    });
    fireEvent.click(button);

    const detail = await screen.findByRole("article", {
      name: /Finding detail: Locked comparison: DATABASE_URL/i,
    });
    expect(
      within(detail).getByLabelText("Masked sensitive value"),
    ).toBeInTheDocument();
    expect(within(detail).queryByText(/postgres:\/\/[^•]/)).not.toBeInTheDocument();
  });
});
