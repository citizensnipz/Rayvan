export type {
  FindingEvaluator,
  FindingEvaluatorInput,
  FindingEvaluatorResult,
} from "./types.js";
export { evaluatorResult } from "./types.js";
export { createConfigurationEvaluator } from "./configuration.js";
export { createEnvironmentEvaluator } from "./environment.js";
export { createIntegrationEvaluator } from "./integration.js";
export { createChangeEvaluator } from "./change.js";

import { createChangeEvaluator } from "./change.js";
import { createConfigurationEvaluator } from "./configuration.js";
import { createEnvironmentEvaluator } from "./environment.js";
import { createIntegrationEvaluator } from "./integration.js";
import type { FindingEvaluator } from "./types.js";

/** Core Rayvan evaluators (configuration, environment/resource, integration, change). */
export function createCoreEvaluators(): FindingEvaluator[] {
  return [
    createConfigurationEvaluator(),
    createEnvironmentEvaluator(),
    createIntegrationEvaluator(),
    createChangeEvaluator(),
  ];
}
