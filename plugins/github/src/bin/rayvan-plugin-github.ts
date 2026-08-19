#!/usr/bin/env node
/**
 * OOP launcher entrypoint for the GitHub plugin.
 * Packaged as `bin/rayvan-plugin-github[.exe]` inside `.rayvan-plugin` archives.
 */
import { serveRayvanPlugin } from "@rayvan/plugin-client";

import { plugin } from "../index.js";

await serveRayvanPlugin(plugin);
