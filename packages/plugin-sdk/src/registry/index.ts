import { supportsCapability } from "../capabilities/index.js";
import {
  PluginNotFoundError,
  PluginValidationError,
} from "../errors/index.js";
import type { PluginCapability, PluginManifest } from "../manifest/index.js";
import type { RayvanPlugin } from "../plugin.js";
import { validatePlugin, validatePluginManifest } from "../validation/index.js";

export interface PluginRegisterOptions {
  /**
   * When true, only the manifest is validated. Use for OOP plugins whose
   * handlers live in a child process (OutOfProcessPluginRuntime).
   */
  handlersExternal?: boolean;
}

export interface PluginRegistry {
  register(plugin: RayvanPlugin, options?: PluginRegisterOptions): void;
  unregister(pluginId: string): void;
  get(pluginId: string): RayvanPlugin | undefined;
  list(): PluginManifest[];
  supports(pluginId: string, capability: PluginCapability): boolean;
}

/**
 * In-process registry for explicitly registered built-in plugins.
 * Does not scan the filesystem or install packages.
 */
export class InProcessPluginRegistry implements PluginRegistry {
  private readonly plugins = new Map<string, RayvanPlugin>();

  register(plugin: RayvanPlugin, options?: PluginRegisterOptions): void {
    if (options?.handlersExternal) {
      validatePluginManifest(plugin.manifest);
    } else {
      validatePlugin(plugin);
    }

    const pluginId = plugin.manifest.id;
    if (this.plugins.has(pluginId)) {
      throw new PluginValidationError(
        `Plugin already registered: ${pluginId}`,
        { pluginId },
      );
    }

    this.plugins.set(pluginId, plugin);
  }

  unregister(pluginId: string): void {
    if (!this.plugins.delete(pluginId)) {
      throw new PluginNotFoundError(pluginId);
    }
  }

  get(pluginId: string): RayvanPlugin | undefined {
    return this.plugins.get(pluginId);
  }

  list(): PluginManifest[] {
    return [...this.plugins.values()].map((plugin) =>
      structuredClone(plugin.manifest),
    );
  }

  supports(pluginId: string, capability: PluginCapability): boolean {
    const plugin = this.plugins.get(pluginId);
    if (!plugin) {
      throw new PluginNotFoundError(pluginId);
    }
    return supportsCapability(plugin.manifest, capability);
  }
}
