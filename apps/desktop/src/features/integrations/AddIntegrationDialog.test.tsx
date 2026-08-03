import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AddIntegrationDialog } from "./AddIntegrationDialog.js";

describe("AddIntegrationDialog", () => {
  it("keeps the file screen across parent re-renders with a new onClose", () => {
    const onSubmit = vi.fn();
    const { rerender } = render(
      <AddIntegrationDialog
        open
        onClose={() => undefined}
        plugins={[]}
        onSubmit={onSubmit}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /Add from file/i }));
    expect(screen.getByLabelText("Package path")).toBeInTheDocument();

    rerender(
      <AddIntegrationDialog
        open
        onClose={() => undefined}
        plugins={[]}
        onSubmit={onSubmit}
      />,
    );

    expect(screen.getByLabelText("Package path")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /Add from library/i }),
    ).not.toBeInTheDocument();
  });
});
