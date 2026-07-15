import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  createDevEnvironmentsGateway,
  type EnvironmentsGateway,
} from "../../lib/environments/index.js";
import { EnvironmentsProvider } from "./EnvironmentsContext.js";
import { EnvironmentsWorkspace } from "./EnvironmentsWorkspace.js";

function createEmptyGateway(): EnvironmentsGateway {
  const real = createDevEnvironmentsGateway();
  return { ...real, ensureProjectSeeded: async () => {} };
}

function renderWorkspace(gateway: EnvironmentsGateway, projectId: string | null) {
  return render(
    <EnvironmentsProvider gateway={gateway}>
      <EnvironmentsWorkspace projectId={projectId} />
    </EnvironmentsProvider>,
  );
}

describe("EnvironmentsWorkspace", () => {
  it("shows empty state when the project has no environments", async () => {
    renderWorkspace(createEmptyGateway(), "project-1");

    expect(await screen.findByText("No environments yet")).toBeInTheDocument();
  });

  it("creates a local_only environment and shows it on the overview", async () => {
    renderWorkspace(createEmptyGateway(), "project-1");
    await screen.findByText("No environments yet");

    fireEvent.click(screen.getByRole("button", { name: "+ Create environment" }));
    const dialog = screen.getByRole("dialog", { name: "Create environment" });

    fireEvent.change(within(dialog).getByLabelText("Name"), {
      target: { value: "QA Lab" },
    });
    fireEvent.change(within(dialog).getByLabelText("Type"), {
      target: { value: "test" },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: "Create environment" }));

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: "Create environment" })).not.toBeInTheDocument();
    });
    expect(await screen.findByText(/QA Lab created as local only/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Overview" }));
    expect(await screen.findByRole("article", { name: /QA Lab/i })).toBeInTheDocument();
    expect(within(screen.getByRole("article", { name: /QA Lab/i })).getByText("Local only")).toBeInTheDocument();
  });

  it("rejects duplicate environment names", async () => {
    const gateway = createEmptyGateway();
    renderWorkspace(gateway, "project-1");
    await screen.findByText("No environments yet");

    fireEvent.click(screen.getByRole("button", { name: "+ Create environment" }));
    let dialog = screen.getByRole("dialog", { name: "Create environment" });
    fireEvent.change(within(dialog).getByLabelText("Name"), {
      target: { value: "Twin" },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: "Create environment" }));
    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: "Create environment" })).not.toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("tab", { name: "Overview" }));
    fireEvent.click(screen.getByRole("button", { name: "+ Create environment" }));
    dialog = screen.getByRole("dialog", { name: "Create environment" });
    fireEvent.change(within(dialog).getByLabelText("Name"), {
      target: { value: "Twin" },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: "Create environment" }));

    expect(await within(dialog).findByRole("alert")).toHaveTextContent(/already exists/i);
  });

  it("seeds fixture environments for a project and scopes by project id", async () => {
    const gateway = createDevEnvironmentsGateway();
    renderWorkspace(gateway, "project-1");

    expect(
      await screen.findByRole("article", { name: /Development/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("article", { name: /Production/i })).toBeInTheDocument();
    expect(screen.getAllByRole("article").length).toBeGreaterThanOrEqual(4);
  });

  it("opens environment and configuration key detail tabs", async () => {
    renderWorkspace(createDevEnvironmentsGateway(), "project-1");
    const article = await screen.findByRole("article", { name: /Development/i });

    fireEvent.click(within(article).getByRole("button", { name: "Open" }));
    expect(await screen.findByRole("tab", { name: "Development" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Configuration Matrix" }));
    const keyButton = await screen.findByRole("button", { name: "API_BASE_URL" });
    fireEvent.click(keyButton);
    expect(await screen.findByRole("tab", { name: "API_BASE_URL" })).toBeInTheDocument();
  });

  it("resets tabs when the current project changes", async () => {
    const gateway = createDevEnvironmentsGateway();
    const { rerender } = renderWorkspace(gateway, "project-1");
    const article = await screen.findByRole("article", { name: /Development/i });

    fireEvent.click(within(article).getByRole("button", { name: "Open" }));
    await screen.findByRole("tab", { name: "Development" });

    rerender(
      <EnvironmentsProvider gateway={gateway}>
        <EnvironmentsWorkspace projectId="project-2" />
      </EnvironmentsProvider>,
    );

    await waitFor(() => {
      expect(screen.queryByRole("tab", { name: "Development" })).not.toBeInTheDocument();
    });
    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  it("accepts and rejects mapping suggestions without auto-bind beforehand", async () => {
    const gateway = createDevEnvironmentsGateway();
    renderWorkspace(gateway, "project-1");
    await screen.findByRole("region", { name: "Mapping suggestions" });

    const section = screen.getByRole("region", { name: "Mapping suggestions" });
    const acceptButtons = within(section).getAllByRole("button", { name: "Accept" });
    const rejectButtons = within(section).getAllByRole("button", { name: "Reject" });
    expect(acceptButtons.length).toBeGreaterThan(0);

    fireEvent.click(rejectButtons[0]!);
    await waitFor(() => {
      expect(screen.getByText(/rejected/i)).toBeInTheDocument();
    });

    const remainingAccept = within(
      screen.getByRole("region", { name: "Mapping suggestions" }),
    ).getAllByRole("button", { name: "Accept" });
    fireEvent.click(remainingAccept[0]!);
    await waitFor(() => {
      expect(screen.getByText(/accepted and resource bound/i)).toBeInTheDocument();
    });
  });

  it("groups resources and supports attach/detach from the Resources tab", async () => {
    renderWorkspace(createDevEnvironmentsGateway(), "project-1");
    await screen.findByRole("article", { name: /Development/i });

    fireEvent.click(screen.getByRole("tab", { name: "Resources" }));
    const resourcesPanel = await screen.findByRole("tabpanel", { name: "Resources" });
    expect(within(resourcesPanel).getByRole("heading", { name: "Unmapped" })).toBeInTheDocument();
    expect(within(resourcesPanel).getByText("GitHub develop")).toBeInTheDocument();

    const unmappedItem = within(resourcesPanel).getByText("GitHub develop").closest("li");
    expect(unmappedItem).toBeTruthy();
    fireEvent.click(within(unmappedItem as HTMLElement).getByRole("button", { name: "Attach" }));

    await waitFor(() => {
      const panel = screen.getByRole("tabpanel", { name: "Resources" });
      const develop = within(panel).getByText("GitHub develop").closest("li");
      expect(
        within(develop as HTMLElement).queryByRole("button", { name: "Detach" }),
      ).toBeInTheDocument();
    });

    const panel = screen.getByRole("tabpanel", { name: "Resources" });
    const boundItem = within(panel).getByText("GitHub develop").closest("li") as HTMLElement;
    fireEvent.click(within(boundItem).getByRole("button", { name: "Detach" }));
    await waitFor(() => {
      const nextPanel = screen.getByRole("tabpanel", { name: "Resources" });
      expect(within(nextPanel).getByRole("heading", { name: "Unmapped" })).toBeInTheDocument();
      const after = within(nextPanel).getByText("GitHub develop").closest("li") as HTMLElement;
      expect(within(after).getByRole("button", { name: "Attach" })).toBeInTheDocument();
    });
  });

  it("archives an environment and removes it from the default list", async () => {
    renderWorkspace(createDevEnvironmentsGateway(), "project-1");
    const article = await screen.findByRole("article", { name: /Local Scratch/i });

    fireEvent.click(within(article).getByRole("button", { name: "More" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "Archive" }));

    await waitFor(() => {
      expect(screen.queryByRole("article", { name: /Local Scratch/i })).not.toBeInTheDocument();
    });
  });
});
