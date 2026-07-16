import type {
  FindingActor,
  FindingLifecycleEventRecord,
  FindingRecord,
} from "@rayvan/core";
import {
  acknowledgeFinding,
  dismissFinding,
  suppressFinding,
  type FindingLifecycleEventRepository,
  type FindingQuery,
  type FindingRepository,
} from "@rayvan/findings-engine";

import {
  FindingNotFoundError,
  FindingPersistenceError,
  InvalidFindingStatusTransitionError,
} from "./errors.js";
import { lifecycleEventToBindParams } from "./mappers.js";
import { SqliteFindingRepository } from "./sqlite-repository.js";
import {
  assertFindingActor,
  resolveSuppressedUntil,
  type FindingSuppressionPreset,
} from "./validation.js";

export type SuppressFindingInput =
  | { until: string; reason?: string }
  | { preset: FindingSuppressionPreset; reason?: string };

export class FindingLifecycleService {
  constructor(
    private readonly findings: FindingRepository,
    private readonly lifecycleEvents: FindingLifecycleEventRepository,
  ) {}

  list(query: FindingQuery): Promise<FindingRecord[]> {
    return this.findings.list(query);
  }

  get(id: string): Promise<FindingRecord | undefined> {
    return this.findings.getById(id);
  }

  listLifecycleEvents(
    findingId: string,
  ): Promise<FindingLifecycleEventRecord[]> {
    return this.lifecycleEvents.listByFindingId(findingId);
  }

  async acknowledge(
    findingId: string,
    actor: FindingActor,
    comment?: string,
    now: string = new Date().toISOString(),
  ): Promise<FindingRecord> {
    assertFindingActor(actor);
    const existing = await this.requireFinding(findingId);
    try {
      const result = acknowledgeFinding(existing, actor, now);
      if (comment) {
        result.event.reason = comment;
      }
      await this.persistMutation(result.record, result.event);
      return result.record;
    } catch (error) {
      throw this.mapLifecycleError(error);
    }
  }

  async dismiss(
    findingId: string,
    actor: FindingActor,
    reason?: string,
    now: string = new Date().toISOString(),
  ): Promise<FindingRecord> {
    assertFindingActor(actor);
    const existing = await this.requireFinding(findingId);
    try {
      const result = dismissFinding(existing, actor, now, reason);
      await this.persistMutation(result.record, result.event);
      return result.record;
    } catch (error) {
      throw this.mapLifecycleError(error);
    }
  }

  async suppress(
    findingId: string,
    actor: FindingActor,
    input: SuppressFindingInput,
    now: string = new Date().toISOString(),
  ): Promise<FindingRecord> {
    assertFindingActor(actor);
    const existing = await this.requireFinding(findingId);
    const suppressedUntil = resolveSuppressedUntil(input, now);
    try {
      const result = suppressFinding(
        existing,
        actor,
        now,
        suppressedUntil,
        input.reason,
      );
      await this.persistMutation(result.record, result.event);
      return result.record;
    } catch (error) {
      throw this.mapLifecycleError(error);
    }
  }

  private async requireFinding(id: string): Promise<FindingRecord> {
    const existing = await this.findings.getById(id);
    if (!existing) {
      throw new FindingNotFoundError(id);
    }
    return existing;
  }

  private async persistMutation(
    record: FindingRecord,
    event: FindingLifecycleEventRecord,
  ): Promise<void> {
    if (this.findings instanceof SqliteFindingRepository) {
      this.findings.saveWithLifecycleEvent(
        record,
        lifecycleEventToBindParams(event),
      );
      return;
    }

    await this.findings.save(record);
    await this.lifecycleEvents.append(event);
  }

  private mapLifecycleError(error: unknown): Error {
    if (
      error instanceof FindingNotFoundError ||
      error instanceof InvalidFindingStatusTransitionError ||
      error instanceof FindingPersistenceError
    ) {
      return error;
    }
    if (error instanceof Error && error.message.startsWith("Cannot acknowledge")) {
      return new InvalidFindingStatusTransitionError(error.message);
    }
    if (error instanceof Error) {
      return new FindingPersistenceError(error.message, error);
    }
    return new FindingPersistenceError("Finding lifecycle mutation failed", error);
  }
}

/** Alias preferred by some call sites. */
export { FindingLifecycleService as FindingService };
