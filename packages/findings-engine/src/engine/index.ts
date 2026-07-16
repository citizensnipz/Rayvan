export { FindingEngine, type FindingEngineDeps } from "./finding-engine.js";
export { matchAndApplyDetections, type DedupeMatchResult } from "./dedupe.js";
export {
  acknowledgeFinding,
  dismissFinding,
  suppressFinding,
  resolveFinding,
  reopenFinding,
  isSuppressionExpired,
  type LifecycleMutationResult,
} from "./lifecycle.js";
