import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { App } from "./App";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async () => []),
}));

describe("Desktop App", () => {
  it("renders the projects experience", async () => {
    render(<App />);

    expect(screen.getByRole("heading", { name: "Rayvan" })).toBeInTheDocument();
    expect(
      screen.getByText("Local-first infrastructure control plane."),
    ).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "Projects" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create project" })).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText("No projects yet.")).toBeInTheDocument();
    });
  });
});
