import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { AddPluginFromFile } from "./AddPluginFromFile.js";

function selectFile(input: HTMLElement, file: File) {
  fireEvent.change(input, { target: { files: [file] } });
}

describe("AddPluginFromFile", () => {
  it("rejects files with an unsupported extension", async () => {
    render(<AddPluginFromFile />);
    const input = screen.getByLabelText("Choose plugin file") as HTMLInputElement;
    const file = new File(["not a plugin"], "malware.exe", { type: "application/octet-stream" });

    selectFile(input, file);

    expect(
      await screen.findByText(
        /Unsupported file type\. Only \.rayvan-plugin \(or metadata \.json\) files are supported\./i,
      ),
    ).toBeInTheDocument();
  });

  it("rejects JSON that does not look like plugin manifest metadata", async () => {
    render(<AddPluginFromFile />);
    const input = screen.getByLabelText("Choose plugin file") as HTMLInputElement;
    const file = new File([JSON.stringify({ hello: "world" })], "not-a-plugin.rayvan-plugin", {
      type: "application/json",
    });

    selectFile(input, file);

    expect(
      await screen.findByText(/does not contain recognizable plugin manifest metadata/i),
    ).toBeInTheDocument();
  });

  it("never executes file contents, even when they look like executable code", async () => {
    const globalWithFlag = globalThis as { __executed?: boolean };
    globalWithFlag.__executed = false;

    render(<AddPluginFromFile />);
    const input = screen.getByLabelText("Choose plugin file") as HTMLInputElement;
    const maliciousSource = "globalThis.__executed = true;";
    const file = new File([maliciousSource], "payload.rayvan-plugin", {
      type: "application/octet-stream",
    });

    selectFile(input, file);

    await screen.findByRole("alert");
    expect(globalWithFlag.__executed).toBe(false);
    delete globalWithFlag.__executed;
  });

  it("shows a development-only message for a recognized manifest without installing or executing it", async () => {
    render(<AddPluginFromFile />);
    const input = screen.getByLabelText("Choose plugin file") as HTMLInputElement;
    const manifest = {
      id: "acme-plugin",
      name: "Acme Plugin",
      version: "1.0.0",
      publisher: "acme",
    };
    const file = new File([JSON.stringify(manifest)], "acme.rayvan-plugin", {
      type: "application/json",
    });

    selectFile(input, file);

    await waitFor(() => {
      expect(screen.getByText(/Detected plugin/i)).toBeInTheDocument();
    });
    expect(screen.getByText("Acme Plugin", { exact: false })).toBeInTheDocument();
    expect(
      screen.getByText(/external plugin execution is not enabled in this build/i),
    ).toBeInTheDocument();
  });
});
