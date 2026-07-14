import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { App } from "./App";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (command: string) => {
    switch (command) {
      case "list_projects":
        return [];
      case "get_current_project_id":
        return null;
      case "set_current_project_id":
        return null;
      default:
        return null;
    }
  }),
}));

describe("Desktop App", () => {
  it("defaults to overview with brand on the left when no project is loaded", async () => {
    render(<App />);

    const topNav = screen.getByRole("banner");
    expect(
      within(topNav).getByRole("heading", { name: "Rayvan" }),
    ).toBeInTheDocument();
    expect(
      within(topNav).getByRole("button", { name: "Create new project" }),
    ).toBeInTheDocument();

    expect(
      await screen.findByText("Create a new project to get started"),
    ).toBeInTheDocument();

    expect(screen.getByRole("button", { name: "Overview" })).not.toBeDisabled();
    expect(screen.getByRole("button", { name: "Overview" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    expect(screen.getByRole("button", { name: "Environments" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Settings" })).not.toBeDisabled();
  });

  it("opens the create project dialog from the top nav", () => {
    render(<App />);

    fireEvent.click(
      within(screen.getByRole("banner")).getByRole("button", {
        name: "Create new project",
      }),
    );

    const dialog = screen.getByRole("dialog", { name: "Create new project" });
    expect(dialog).toBeInTheDocument();
    expect(
      within(dialog).getByRole("button", { name: "Create project" }),
    ).toBeInTheDocument();
  });

  it("shows only the dark mode toggle on Settings", async () => {
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Settings" }));

    expect(screen.getByRole("heading", { name: "Settings" })).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Dark mode" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Projects" })).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Create project" }),
    ).not.toBeInTheDocument();
    expect(
      await screen.findByRole("button", { name: "Create new project" }),
    ).toBeInTheDocument();
  });

  it("toggles dark mode from Settings", () => {
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Settings" }));

    const toggle = screen.getByRole("switch", { name: "Dark mode" });
    expect(toggle).toHaveAttribute("aria-checked", "false");
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");

    fireEvent.click(toggle);

    expect(toggle).toHaveAttribute("aria-checked", "true");
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
    expect(screen.getByText("Using dark appearance")).toBeInTheDocument();
  });
});
