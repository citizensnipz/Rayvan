export type {
  AcceptSuggestionInput,
  AcceptSuggestionResult,
  AdoptDiscoveredKeyGatewayInput,
  AttachResourceInput,
  CreateEnvironmentGatewayInput,
  EnvironmentSyncPhase,
  EnvironmentSyncPluginResult,
  EnvironmentSyncResult,
  EnvironmentSyncState,
  EnvironmentsGateway,
  MoveResourceInput,
  SaveDesiredValueGatewayInput,
  UpdateConfigurationKeyGatewayInput,
  UpdateEnvironmentGatewayInput,
} from "./types.js";

export { createDevEnvironmentsGateway } from "./dev-gateway.js";
export { createDaemonEnvironmentsGateway } from "./daemon-gateway.js";
export {
  ENVIRONMENTS_FIXTURE_ACTOR,
  seedProjectEnvironments,
} from "./dev-fixtures.js";
