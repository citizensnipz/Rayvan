import { randomUUID } from "node:crypto";

import { PluginDomainError, PluginNotFoundError } from "../errors.js";
import type {
  PluginSetupAuthMethod,
  PluginSetupSessionRecord,
} from "../models-setup.js";
import type { InstalledPluginRepository } from "../repositories/types.js";
import type { PluginSetupSessionRepository } from "../repositories/setup-sessions.js";
import { sanitizePluginSetupStatePatch } from "../secrets.js";

export class PluginSetupSessionService {
  constructor(
    private readonly installedPlugins: InstalledPluginRepository,
    private readonly sessions: PluginSetupSessionRepository,
  ) {}

  async start(input: {
    pluginId: string;
    projectId?: string;
    authMethod?: PluginSetupAuthMethod;
  }): Promise<PluginSetupSessionRecord> {
    const installed = await this.installedPlugins.getByPluginId(input.pluginId);
    if (!installed || !installed.enabled) {
      throw new PluginNotFoundError(input.pluginId);
    }
    const now = new Date().toISOString();
    const steps = installed.manifestSnapshot.setup?.steps ?? [];
    const record: PluginSetupSessionRecord = {
      id: randomUUID(),
      pluginId: input.pluginId,
      installedPluginId: installed.id,
      projectId: input.projectId,
      status: "active",
      currentStepId: steps[0]?.id,
      authMethod: input.authMethod,
      state: {},
      createdAt: now,
      updatedAt: now,
    };
    await this.sessions.save(record);
    return record;
  }

  async step(input: {
    sessionId: string;
    stepId: string;
    /** Non-secret patch merged into session.state. */
    statePatch?: Record<string, unknown>;
    authMethod?: PluginSetupAuthMethod;
  }): Promise<PluginSetupSessionRecord> {
    const existing = await this.sessions.getById(input.sessionId);
    if (!existing || existing.status !== "active") {
      throw new PluginDomainError(`Setup session not active: ${input.sessionId}`);
    }
    const now = new Date().toISOString();
    const statePatch = input.statePatch
      ? sanitizePluginSetupStatePatch(input.statePatch)
      : undefined;
    const next: PluginSetupSessionRecord = {
      ...existing,
      currentStepId: input.stepId,
      authMethod: input.authMethod ?? existing.authMethod,
      state: {
        ...existing.state,
        ...(statePatch ?? {}),
      },
      updatedAt: now,
    };
    await this.sessions.save(next);
    return next;
  }

  async complete(sessionId: string): Promise<PluginSetupSessionRecord> {
    const existing = await this.sessions.getById(sessionId);
    if (!existing || existing.status !== "active") {
      throw new PluginDomainError(`Setup session not active: ${sessionId}`);
    }
    const now = new Date().toISOString();
    const next: PluginSetupSessionRecord = {
      ...existing,
      status: "completed",
      updatedAt: now,
      completedAt: now,
    };
    await this.sessions.save(next);
    return next;
  }

  async get(sessionId: string): Promise<PluginSetupSessionRecord | undefined> {
    return this.sessions.getById(sessionId);
  }
}
