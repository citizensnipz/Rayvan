export * from "./errors.js";
export * from "./validation.js";
export * from "./repository.js";
export * from "./mappers.js";
export * from "./memory-repository.js";
export * from "./sqlite-repository.js";
export * from "./lifecycle-repository.js";
export * from "./evaluation-run-repository.js";
export * from "./service.js";
export * from "./summary-service.js";

import {
  InMemoryFindingEvaluationRunRepository,
} from "./evaluation-run-repository.js";
import { InMemoryFindingLifecycleEventRepository } from "./lifecycle-repository.js";
import { InMemoryFindingRepository } from "./memory-repository.js";
import { FindingLifecycleService } from "./service.js";
import { FindingSummaryService } from "./summary-service.js";

export interface InMemoryFindingsPersistence {
  findings: InMemoryFindingRepository;
  lifecycleEvents: InMemoryFindingLifecycleEventRepository;
  evaluationRuns: InMemoryFindingEvaluationRunRepository;
  service: FindingLifecycleService;
  summaryService: FindingSummaryService;
}

export function createInMemoryFindingsPersistence(): InMemoryFindingsPersistence {
  const findings = new InMemoryFindingRepository();
  const lifecycleEvents = new InMemoryFindingLifecycleEventRepository();
  const evaluationRuns = new InMemoryFindingEvaluationRunRepository();
  return {
    findings,
    lifecycleEvents,
    evaluationRuns,
    service: new FindingLifecycleService(findings, lifecycleEvents),
    summaryService: new FindingSummaryService(findings),
  };
}
