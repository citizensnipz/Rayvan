import {
  NotImplementedPluginError,
  type RayvanPlugin,
} from "@rayvan/plugin-sdk";
import { manifest } from "./manifest.js";

function notImplemented(phase: string): never {
  throw new NotImplementedPluginError(phase);
}

export const plugin: RayvanPlugin = {
  manifest,

  async initialize() {
    notImplemented("initialize");
  },

  async testConnection() {
    notImplemented("connect");
  },

  async discoverResources() {
    notImplemented("discover");
  },

  async collectConfiguration() {
    notImplemented("inspect");
  },

  async collectHealth() {
    notImplemented("inspect");
  },

  async planAction() {
    notImplemented("plan");
  },

  async executeAction() {
    notImplemented("execute");
  },

  async dispose() {
    return;
  },
};

export { manifest };
export default plugin;
