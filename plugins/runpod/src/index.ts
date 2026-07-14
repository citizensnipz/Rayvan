import type { RayvanPlugin } from "@rayvan/plugin-sdk";
import { manifest } from "./manifest.js";

/** Placeholder plugin. Capabilities and handlers will be added with the real integration. */
export const plugin: RayvanPlugin = {
  manifest,
};

export { manifest };
export default plugin;
