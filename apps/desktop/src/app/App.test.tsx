import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { App } from "./App";

describe("Desktop App", () => {
  it("renders the initial empty state", () => {
    render(<App />);

    expect(screen.getByRole("heading", { name: "Rayvan" })).toBeInTheDocument();
    expect(
      screen.getByText("Local-first infrastructure control plane."),
    ).toBeInTheDocument();
    expect(screen.getByText("No projects connected yet.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add project" })).toBeDisabled();
  });
});
