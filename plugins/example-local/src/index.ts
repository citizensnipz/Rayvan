import {
  PluginExecutionError,
  type ApplyContext,
  type ChangePlan,
  type DiscoveryContext,
  type EvaluateFindingsContext,
  type EvaluateFindingsResult,
  type InspectContext,
  type ObservedResourceState,
  type PlanContext,
  type RayvanPlugin,
  type VerifyContext,
  validateApplyResult,
  validateChangePlan,
  validateDiscoveredResource,
  validateEvaluateFindingsResult,
  validateVerificationResult,
} from "@rayvan/plugin-sdk";
import {
  EXAMPLE_LOCAL_PLUGIN_ID,
  LOCAL_SERVICE_RESOURCE_TYPE,
  LOCAL_SERVICE_SCHEMA_VERSION,
  manifest,
} from "./manifest.js";
import { ExampleLocalStore } from "./store.js";

const store = new ExampleLocalStore();

function requireLocalService(
  providerResourceId: string,
  capability: "inspect" | "plan" | "apply" | "verify",
) {
  const service = store.get(providerResourceId);
  if (!service) {
    throw new PluginExecutionError(
      EXAMPLE_LOCAL_PLUGIN_ID,
      capability,
      `Unknown local service: ${providerResourceId}`,
    );
  }
  return service;
}

function toObservedState(
  resourceId: string,
  providerResourceId: string,
  port: number,
  status: "ready" | "degraded",
): ObservedResourceState {
  return {
    resourceId,
    pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
    resourceType: LOCAL_SERVICE_RESOURCE_TYPE,
    observedAt: "1970-01-01T00:00:00.000Z",
    status,
    attributes: {
      port,
      providerResourceId,
      schemaVersion: LOCAL_SERVICE_SCHEMA_VERSION,
    },
    checks: [
      {
        name: "port-configured",
        status: "pass",
        message: `Listening on ${port}`,
      },
    ],
  };
}

async function discover(context: DiscoveryContext) {
  if (context.pluginId !== EXAMPLE_LOCAL_PLUGIN_ID) {
    throw new PluginExecutionError(
      EXAMPLE_LOCAL_PLUGIN_ID,
      "discover",
      `Unexpected plugin id: ${context.pluginId}`,
    );
  }

  const discovered = store.list().map((service) => store.toDiscovered(service));
  for (const resource of discovered) {
    validateDiscoveredResource(resource, EXAMPLE_LOCAL_PLUGIN_ID);
  }
  return discovered;
}

async function inspect(context: InspectContext): Promise<ObservedResourceState> {
  if (context.resource.resourceType !== LOCAL_SERVICE_RESOURCE_TYPE) {
    throw new PluginExecutionError(
      EXAMPLE_LOCAL_PLUGIN_ID,
      "inspect",
      `Unsupported resource type: ${context.resource.resourceType}`,
    );
  }

  const service = requireLocalService(
    context.resource.providerResourceId,
    "inspect",
  );
  return toObservedState(
    context.resource.resourceId,
    service.providerResourceId,
    service.port,
    service.status,
  );
}

async function plan(context: PlanContext): Promise<ChangePlan> {
  const service = requireLocalService(
    context.resource.providerResourceId,
    "plan",
  );
  const desiredPort = context.desired.attributes.port;

  if (typeof desiredPort !== "number" || !Number.isInteger(desiredPort)) {
    throw new PluginExecutionError(
      EXAMPLE_LOCAL_PLUGIN_ID,
      "plan",
      "desired.attributes.port must be an integer",
    );
  }

  const operations =
    desiredPort === service.port
      ? []
      : [
          {
            id: "set-port",
            type: "update_attribute",
            description: `Change port from ${service.port} to ${desiredPort}`,
            path: "attributes.port",
            before: service.port,
            after: desiredPort,
            requiresApproval: true,
            destructive: false,
          },
        ];

  const changePlan: ChangePlan = {
    id: `plan-${context.resource.providerResourceId}-${desiredPort}`,
    pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
    resourceId: context.resource.resourceId,
    summary:
      operations.length === 0
        ? "No changes required"
        : `Update ${service.name} port to ${desiredPort}`,
    operations,
    warnings: [],
    destructive: false,
  };

  validateChangePlan(changePlan);
  return changePlan;
}

async function apply(context: ApplyContext) {
  const { plan: approvedPlan } = context.approvedPlan;
  if (approvedPlan.pluginId !== EXAMPLE_LOCAL_PLUGIN_ID) {
    throw new PluginExecutionError(
      EXAMPLE_LOCAL_PLUGIN_ID,
      "apply",
      `Unexpected plugin id on approved plan: ${approvedPlan.pluginId}`,
    );
  }

  if (context.resource.resourceId !== approvedPlan.resourceId) {
    throw new PluginExecutionError(
      EXAMPLE_LOCAL_PLUGIN_ID,
      "apply",
      "Approved plan resourceId does not match apply context binding",
    );
  }

  const providerResourceId = context.resource.providerResourceId;
  const serviceBefore = requireLocalService(providerResourceId, "apply");
  const appliedOperationIds: string[] = [];

  for (const operation of approvedPlan.operations) {
    if (
      operation.type !== "update_attribute" ||
      operation.path !== "attributes.port"
    ) {
      throw new PluginExecutionError(
        EXAMPLE_LOCAL_PLUGIN_ID,
        "apply",
        `Unsupported operation: ${operation.type}`,
      );
    }
    if (typeof operation.after !== "number") {
      throw new PluginExecutionError(
        EXAMPLE_LOCAL_PLUGIN_ID,
        "apply",
        "Port operation is missing a numeric after value",
      );
    }

    store.setPort(providerResourceId, operation.after);
    appliedOperationIds.push(operation.id);
  }

  const service = store.get(providerResourceId) ?? serviceBefore;

  const result = {
    ok: true,
    message:
      appliedOperationIds.length === 0
        ? "No operations to apply"
        : `Applied ${appliedOperationIds.length} operation(s)`,
    appliedOperationIds,
    resultingState: toObservedState(
      approvedPlan.resourceId,
      providerResourceId,
      service.port,
      service.status,
    ),
  };
  validateApplyResult(result, EXAMPLE_LOCAL_PLUGIN_ID);
  return result;
}

async function verify(context: VerifyContext) {
  const service = requireLocalService(
    context.resource.providerResourceId,
    "verify",
  );
  const observed = toObservedState(
    context.resource.resourceId,
    service.providerResourceId,
    service.port,
    service.status,
  );
  const desiredPort = context.approvedPlan.plan.operations.find(
    (operation) => operation.path === "attributes.port",
  )?.after;

  const result =
    typeof desiredPort === "number" && service.port !== desiredPort
      ? {
          ok: false,
          message: `Expected port ${desiredPort}, observed ${service.port}`,
          observed,
          mismatches: [
            `attributes.port: expected ${desiredPort}, got ${service.port}`,
          ],
        }
      : {
          ok: true,
          message: "Local service matches approved plan",
          observed,
          mismatches: [],
        };

  validateVerificationResult(result, EXAMPLE_LOCAL_PLUGIN_ID);
  return result;
}

async function evaluateFindings(
  context: EvaluateFindingsContext,
): Promise<EvaluateFindingsResult> {
  if (context.pluginId !== EXAMPLE_LOCAL_PLUGIN_ID) {
    throw new PluginExecutionError(
      EXAMPLE_LOCAL_PLUGIN_ID,
      "evaluate_findings",
      `Unexpected plugin id: ${context.pluginId}`,
    );
  }

  const services = store.list();
  const result: EvaluateFindingsResult =
    services.length === 0
      ? { detections: [], warnings: [] }
      : {
          detections: [
            {
              ruleId: `${EXAMPLE_LOCAL_PLUGIN_ID}.service-present`,
              severity: "info",
              title: "Local services available",
              summary: `${services.length} mock local service(s) are present`,
              scope: {},
              evidence: [
                {
                  type: "message",
                  message: `Discovered ${services.map((service) => service.providerResourceId).join(", ")}`,
                },
              ],
              fingerprintParts: [
                EXAMPLE_LOCAL_PLUGIN_ID,
                "service-present",
                context.connectionId,
              ],
              remediation: {
                type: "manual",
                label: "No action required",
                instructions:
                  "Informational only — demonstrates evaluate_findings detections.",
              },
            },
          ],
          warnings: [],
        };

  validateEvaluateFindingsResult(result, EXAMPLE_LOCAL_PLUGIN_ID);
  return result;
}

export const plugin: RayvanPlugin = {
  manifest,
  discover,
  inspect,
  plan,
  apply,
  verify,
  evaluateFindings,
};

/** Test helper to restore deterministic fixtures between cases. */
export function resetExampleLocalStore(): void {
  store.reset();
}

export { manifest, EXAMPLE_LOCAL_PLUGIN_ID, LOCAL_SERVICE_RESOURCE_TYPE };
export default plugin;
