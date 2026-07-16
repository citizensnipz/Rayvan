import type { FindingActor } from "@rayvan/core";
import { isFindingStatus } from "@rayvan/core";

export type FindingSuppressionPreset = "1h" | "24h" | "7d" | "30d";

const PRESET_MS: Record<FindingSuppressionPreset, number> = {
  "1h": 60 * 60 * 1000,
  "24h": 24 * 60 * 60 * 1000,
  "7d": 7 * 24 * 60 * 60 * 1000,
  "30d": 30 * 24 * 60 * 60 * 1000,
};

export function assertFindingActor(actor: FindingActor): FindingActor {
  if (!actor || typeof actor !== "object" || !("kind" in actor)) {
    throw new Error("Finding actor is required");
  }
  return actor;
}

export function resolveSuppressedUntil(
  input: { until: string } | { preset: FindingSuppressionPreset },
  now: string = new Date().toISOString(),
): string {
  if ("until" in input) {
    const untilMs = Date.parse(input.until);
    if (Number.isNaN(untilMs)) {
      throw new Error(`Invalid suppressedUntil timestamp: ${input.until}`);
    }
    return new Date(untilMs).toISOString();
  }

  const nowMs = Date.parse(now);
  if (Number.isNaN(nowMs)) {
    throw new Error(`Invalid now timestamp: ${now}`);
  }
  return new Date(nowMs + PRESET_MS[input.preset]).toISOString();
}

export function assertOptionalFindingStatus(
  value: string | undefined,
): void {
  if (value !== undefined && !isFindingStatus(value)) {
    throw new Error(`Invalid finding status: ${value}`);
  }
}
