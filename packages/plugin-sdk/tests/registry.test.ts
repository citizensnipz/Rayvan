import { describe, expect, it } from "vitest";
import {
  InProcessPluginRegistry,
  PluginNotFoundError,
  PluginValidationError,
  PluginVersionError,
  RAYVAN_PLUGIN_API_VERSION,
  type PluginManifest,
  type RayvanPlugin,
} from "../src/index.js";

function baseManifest(
  overrides: Partial<PluginManifest> = {},
): PluginManifest {
  return {
    id: "demo",
    name: "Demo",
    version: "1.0.0",
    publisher: "rayvan",
    rayvanApiVersion: RAYVAN_PLUGIN_API_VERSION,
    capabilities: [],
    permissions: [],
    resourceTypes: [],
    ...overrides,
  };
}

describe("InProcessPluginRegistry", () => {
  it("registers a plugin and lists its manifest", () => {
    const registry = new InProcessPluginRegistry();
    const plugin: RayvanPlugin = { manifest: baseManifest() };

    registry.register(plugin);

    expect(registry.get("demo")).toBe(plugin);
    expect(registry.list()).toEqual([plugin.manifest]);
  });

  it("rejects duplicate plugin ids", () => {
    const registry = new InProcessPluginRegistry();
    const plugin: RayvanPlugin = { manifest: baseManifest() };

    registry.register(plugin);

    expect(() => registry.register({ manifest: baseManifest() })).toThrow(
      PluginValidationError,
    );
  });

  it("rejects invalid manifests", () => {
    const registry = new InProcessPluginRegistry();

    expect(() =>
      registry.register({
        manifest: baseManifest({ id: "Invalid_ID" }),
      }),
    ).toThrow(PluginValidationError);

    expect(() =>
      registry.register({
        manifest: baseManifest({ version: "not-semver" }),
      }),
    ).toThrow(PluginValidationError);

    expect(() =>
      registry.register({
        manifest: baseManifest({ rayvanApiVersion: "999" }),
      }),
    ).toThrow(PluginVersionError);
  });

  it("rejects handler/capability mismatches", () => {
    const registry = new InProcessPluginRegistry();

    expect(() =>
      registry.register({
        manifest: baseManifest({ capabilities: ["discover"] }),
      }),
    ).toThrow(/discover/);

    expect(() =>
      registry.register({
        manifest: baseManifest(),
        async discover() {
          return [];
        },
      }),
    ).toThrow(/discover/);
  });

  it("reports capability support and unregisters plugins", () => {
    const registry = new InProcessPluginRegistry();
    registry.register({
      manifest: baseManifest({
        id: "capable",
        capabilities: ["discover"],
      }),
      async discover() {
        return [];
      },
    });

    expect(registry.supports("capable", "discover")).toBe(true);
    expect(registry.supports("capable", "apply")).toBe(false);

    registry.unregister("capable");
    expect(registry.get("capable")).toBeUndefined();
    expect(() => registry.supports("capable", "discover")).toThrow(
      PluginNotFoundError,
    );
  });

  it("rejects duplicate capabilities and permissions", () => {
    const registry = new InProcessPluginRegistry();

    expect(() =>
      registry.register({
        manifest: baseManifest({
          capabilities: ["discover", "discover"],
        }),
        async discover() {
          return [];
        },
      }),
    ).toThrow(/Duplicate capability/);

    expect(() =>
      registry.register({
        manifest: baseManifest({
          permissions: ["network", "network"],
        }),
      }),
    ).toThrow(/Duplicate permission/);
  });
});
