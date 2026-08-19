import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AddPluginFromFile } from "./AddPluginFromFile.js";

const installPluginFromPath = vi.fn();

vi.mock("../../lib/daemon/client.js", () => ({
  desktopDaemon: {
    installPluginFromPath: (...args: unknown[]) =>
      installPluginFromPath(...args),
  },
}));

describe("AddPluginFromFile", () => {
  beforeEach(() => {
    installPluginFromPath.mockReset();
  });

  it("requires a package path before installing", async () => {
    render(<AddPluginFromFile />);
    fireEvent.click(screen.getByRole("button", { name: /Install package/i }));
    expect(
      await screen.findByText(/Enter the absolute path/i),
    ).toBeInTheDocument();
    expect(installPluginFromPath).not.toHaveBeenCalled();
  });

  it("installs via the daemon and offers setup", async () => {
    const onInstalled = vi.fn();
    installPluginFromPath.mockResolvedValue({
      pluginId: "io.rayvan.github",
      version: "0.1.0",
      trustLabel: "Unsigned (development)",
    });

    render(<AddPluginFromFile onInstalled={onInstalled} />);
    fireEvent.change(screen.getByLabelText("Package path"), {
      target: {
        value:
          "C:\\plugins\\io.rayvan.github-0.1.0-x86_64-pc-windows-msvc.rayvan-plugin",
      },
    });
    fireEvent.click(screen.getByRole("button", { name: /Install package/i }));

    await waitFor(() => {
      expect(installPluginFromPath).toHaveBeenCalledWith(
        "C:\\plugins\\io.rayvan.github-0.1.0-x86_64-pc-windows-msvc.rayvan-plugin",
      );
    });
    expect(await screen.findByText(/Installed/i)).toBeInTheDocument();
    expect(screen.getByText(/Unsigned \(development\)/i)).toBeInTheDocument();
    expect(onInstalled).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Set up GitHub/i }));
    expect(onInstalled).toHaveBeenCalledTimes(2);
  });

  it("surfaces daemon install errors without executing package contents", async () => {
    installPluginFromPath.mockRejectedValue(
      new Error("Signature verification failed"),
    );

    render(<AddPluginFromFile />);
    fireEvent.change(screen.getByLabelText("Package path"), {
      target: { value: "D:\\bad.rayvan-plugin" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Install package/i }));

    expect(
      await screen.findByText(/Signature verification failed/i),
    ).toBeInTheDocument();
  });
});
