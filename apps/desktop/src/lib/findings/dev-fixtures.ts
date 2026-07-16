import {
  FINDING_FINGERPRINT_VERSION,
  FINDING_SCHEMA_VERSION,
  configurationKeyId,
  environmentId,
  findingEvaluationRunId,
  findingId,
  findingLifecycleEventId,
  projectId,
  type FindingLifecycleEventRecord,
  type FindingRecord,
  type FindingRemediation,
  type FindingSeverity,
  type FindingStatus,
} from "@rayvan/core";
import {
  CORE_FINDING_RULE_IDS,
  type ProjectFindingsContext,
} from "@rayvan/findings-engine";
import type { InMemoryFindingsPersistence } from "@rayvan/local-database";

import {
  DEV_FINDINGS_ENGINE_ACTOR,
  DEV_FINDINGS_SYSTEM_ACTOR,
  type FindingsSeedContext,
} from "./types.js";

/**
 * DEVELOPMENT ONLY fake findings for the Findings workspace.
 * Navigate to Findings with a project selected — ensureProjectSeeded auto-seeds these.
 * “Scan project” runs evaluateProject against a minimal seeded context.
 */

const NOW = "2026-07-16T10:00:00.000Z";
const EARLIER = "2026-07-15T08:30:00.000Z";
const RECENT = "2026-07-16T11:45:00.000Z";
const RUN_ID = findingEvaluationRunId("dev-findings-run-1");

function envId(
  project: string,
  name: string,
  override?: Record<string, string>,
): ReturnType<typeof environmentId> {
  const fromContext = override?.[name];
  return environmentId(fromContext ?? `${project}:env:${slug(name)}`);
}

function slug(name: string): string {
  return name.toLowerCase().replace(/\s+/g, "-");
}

function connId(
  project: string,
  pluginId: string,
  override?: Record<string, string>,
): string {
  return override?.[pluginId] ?? `${project}:conn:${pluginId}`;
}

interface FixtureSpec {
  idSuffix: string;
  ruleId: string;
  source: FindingRecord["source"];
  category: FindingRecord["category"];
  severity: FindingSeverity;
  title: string;
  summary: string;
  description?: string;
  status?: FindingStatus;
  environmentName?: string;
  pluginId?: string;
  configurationKeyName?: string;
  evidence: FindingRecord["evidence"];
  remediation?: FindingRemediation;
  firstDetectedAt?: string;
  lastDetectedAt?: string;
  occurrenceCount?: number;
  metadata?: Record<string, unknown>;
  dismissalReason?: string;
}

function buildRecord(
  project: string,
  spec: FixtureSpec,
  context?: FindingsSeedContext,
): FindingRecord {
  const environment =
    spec.environmentName !== undefined
      ? envId(project, spec.environmentName, context?.environmentIdsByName)
      : undefined;
  const connectionId = spec.pluginId
    ? connId(project, spec.pluginId, context?.connectionIdsByPluginId)
    : undefined;

  return {
    id: findingId(`${project}:finding:${spec.idSuffix}`),
    projectId: projectId(project),
    ruleId: spec.ruleId,
    source: spec.source,
    category: spec.category,
    severity: spec.severity,
    title: spec.title,
    summary: spec.summary,
    description: spec.description,
    status: spec.status ?? "open",
    fingerprint: `fp:${project}:${spec.idSuffix}`,
    fingerprintVersion: FINDING_FINGERPRINT_VERSION,
    environmentId: environment,
    connectionId,
    configurationKeyId: spec.configurationKeyName
      ? configurationKeyId(`${project}:key:${spec.configurationKeyName}`)
      : undefined,
    evidence: spec.evidence,
    remediation: spec.remediation,
    firstDetectedAt: spec.firstDetectedAt ?? EARLIER,
    lastDetectedAt: spec.lastDetectedAt ?? NOW,
    occurrenceCount: spec.occurrenceCount ?? 1,
    dismissalReason: spec.dismissalReason,
    lastEvaluationRunId: RUN_ID,
    metadata: {
      ...(spec.metadata ?? {}),
      ...(spec.source.type === "plugin" ? { mockPluginFinding: true } : {}),
    },
    schemaVersion: FINDING_SCHEMA_VERSION,
  };
}

function coreFixtures(project: string, context?: FindingsSeedContext): FindingRecord[] {
  const specs: FixtureSpec[] = [
    {
      idSuffix: "missing-required",
      ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_MISSING_REQUIRED,
      source: { type: "rayvan" },
      category: "configuration",
      severity: "critical",
      title: "Missing required configuration: STRIPE_SECRET_KEY",
      summary: "Production is missing a required secret key.",
      description:
        "STRIPE_SECRET_KEY is marked required for Production but no desired or observed value is present.",
      environmentName: "Production",
      configurationKeyName: "STRIPE_SECRET_KEY",
      evidence: [
        {
          type: "configuration_comparison",
          configurationKeyId: `${project}:key:STRIPE_SECRET_KEY`,
          environmentId: String(
            envId(project, "Production", context?.environmentIdsByName),
          ),
          observedStates: [],
        },
      ],
      remediation: {
        type: "open_environment",
        environmentId: String(
          envId(project, "Production", context?.environmentIdsByName),
        ),
        label: "Open Production environment",
      },
      lastDetectedAt: RECENT,
      occurrenceCount: 3,
    },
    {
      idSuffix: "mismatch",
      ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
      source: { type: "rayvan" },
      category: "drift",
      severity: "error",
      title: "Configuration mismatch: API_BASE_URL",
      summary: "Desired and observed values disagree in Production.",
      environmentName: "Production",
      configurationKeyName: "API_BASE_URL",
      evidence: [
        {
          type: "configuration_comparison",
          configurationKeyId: `${project}:key:API_BASE_URL`,
          environmentId: String(
            envId(project, "Production", context?.environmentIdsByName),
          ),
          expectedState: {
            access: "readable",
            value: "https://api.example.rayvan.com",
            sensitive: false,
          },
          observedStates: [
            {
              pluginId: "vercel",
              label: "Vercel Production",
              value: {
                access: "readable",
                value: "https://api.staging.example.rayvan.com",
                sensitive: false,
              },
              inSync: false,
              observedAt: NOW,
            },
          ],
        },
      ],
      remediation: {
        type: "generate_change_plan",
        pluginId: "vercel",
        configurationKeyIds: [`${project}:key:API_BASE_URL`],
        label: "Generate change plan",
      },
      lastDetectedAt: RECENT,
    },
    {
      idSuffix: "unapplied",
      ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_UNAPPLIED,
      source: { type: "rayvan" },
      category: "configuration",
      severity: "warning",
      title: "Unapplied configuration: NODE_ENV",
      summary: "Local desired value has not been applied to Staging.",
      environmentName: "Staging",
      configurationKeyName: "NODE_ENV",
      evidence: [
        {
          type: "message",
          message: "Desired revision is ahead of applied state.",
        },
      ],
      remediation: {
        type: "open_environment",
        environmentId: String(
          envId(project, "Staging", context?.environmentIdsByName),
        ),
        label: "Review Staging configuration",
      },
    },
    {
      idSuffix: "remote-changed",
      ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_REMOTE_CHANGED,
      source: { type: "rayvan" },
      category: "drift",
      severity: "warning",
      title: "Remote value changed: DEBUG_MODE",
      summary: "Observed remote value changed outside Rayvan.",
      environmentName: "Development",
      configurationKeyName: "DEBUG_MODE",
      evidence: [
        {
          type: "configuration_comparison",
          configurationKeyId: `${project}:key:DEBUG_MODE`,
          environmentId: String(
            envId(project, "Development", context?.environmentIdsByName),
          ),
          expectedState: {
            access: "readable",
            value: "false",
            sensitive: false,
          },
          observedStates: [
            {
              pluginId: "vercel",
              label: "Vercel Development",
              value: { access: "readable", value: "true", sensitive: false },
              inSync: false,
              observedAt: NOW,
            },
          ],
        },
      ],
      remediation: {
        type: "resync",
        pluginId: "vercel",
        label: "Resync Development",
      },
    },
    {
      idSuffix: "partially-applied",
      ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_PARTIALLY_APPLIED,
      source: { type: "rayvan" },
      category: "configuration",
      severity: "warning",
      title: "Partially applied configuration change",
      summary: "Some apply operations succeeded; others failed in Preview.",
      environmentName: "Preview",
      evidence: [
        {
          type: "message",
          message: "2 of 3 configuration operations applied successfully.",
        },
      ],
      remediation: {
        type: "manual",
        label: "Review partial apply",
        instructions:
          "Open the environment apply history and retry failed operations after fixing provider errors.",
      },
    },
    {
      idSuffix: "locked-comparison",
      ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_COMPARISON_UNAVAILABLE,
      source: { type: "rayvan" },
      category: "configuration",
      severity: "info",
      title: "Locked comparison: DATABASE_URL",
      summary: "Sensitive value is locked; comparison is unavailable.",
      environmentName: "Production",
      configurationKeyName: "DATABASE_URL",
      evidence: [
        {
          type: "configuration_comparison",
          configurationKeyId: `${project}:key:DATABASE_URL`,
          environmentId: String(
            envId(project, "Production", context?.environmentIdsByName),
          ),
          expectedState: {
            access: "masked",
            maskedValue: "postgres://••••@db.prod.example.rayvan.dev:5432/rayvan",
            sensitive: true,
          },
          observedStates: [
            {
              pluginId: "supabase",
              label: "Supabase Production",
              value: {
                access: "locked",
                sensitive: true,
              },
              observedAt: NOW,
            },
          ],
        },
      ],
      remediation: {
        type: "manual",
        label: "Unlock to compare",
        instructions:
          "Reveal or unlock the secret in the provider console if comparison is required. Rayvan never stores plaintext secrets.",
      },
    },
    {
      idSuffix: "unmapped-resource",
      ruleId: CORE_FINDING_RULE_IDS.RESOURCE_UNMAPPED,
      source: { type: "rayvan" },
      category: "mapping",
      severity: "warning",
      title: "Unmapped resource discovered",
      summary: "A discovered resource is not bound to an environment.",
      evidence: [
        {
          type: "resource_state",
          resourceBindingId: `${project}:resource:unmapped`,
          state: "unmapped",
          observedAt: NOW,
        },
      ],
      remediation: {
        type: "open_environment",
        environmentId: String(
          envId(project, "Development", context?.environmentIdsByName),
        ),
        label: "Open mapping suggestions",
      },
    },
    {
      idSuffix: "connection-expired",
      ruleId: CORE_FINDING_RULE_IDS.INTEGRATION_CONNECTION_EXPIRED,
      source: { type: "rayvan" },
      category: "integration",
      severity: "error",
      title: "Connection expired: GitHub",
      summary: "The GitHub connection credentials have expired.",
      pluginId: "github",
      evidence: [
        {
          type: "connection_error",
          connectionId: connId(
            project,
            "github",
            context?.connectionIdsByPluginId,
          ),
          errorCode: "credential_expired",
          safeMessage: "Authentication token expired. Reauthenticate to continue sync.",
        },
      ],
      remediation: {
        type: "reauthenticate",
        connectionId: connId(
          project,
          "github",
          context?.connectionIdsByPluginId,
        ),
        label: "Reauthenticate GitHub",
      },
    },
    {
      idSuffix: "permission-missing",
      ruleId: CORE_FINDING_RULE_IDS.INTEGRATION_PERMISSION_MISSING,
      source: { type: "rayvan" },
      category: "permission",
      severity: "warning",
      title: "Permission missing: Vercel env write",
      summary: "The Vercel connection lacks permission to write environment variables.",
      pluginId: "vercel",
      evidence: [
        {
          type: "connection_error",
          connectionId: connId(
            project,
            "vercel",
            context?.connectionIdsByPluginId,
          ),
          errorCode: "permission_missing",
          safeMessage: "Missing permission: environment:write",
        },
      ],
      remediation: {
        type: "open_integration",
        connectionId: connId(
          project,
          "vercel",
          context?.connectionIdsByPluginId,
        ),
        label: "Open Vercel integration",
      },
    },
    {
      idSuffix: "apply-failed",
      ruleId: CORE_FINDING_RULE_IDS.CHANGE_APPLY_FAILED,
      source: { type: "rayvan" },
      category: "configuration",
      severity: "error",
      title: "Apply failed: Staging configuration",
      summary: "A configuration apply plan failed for Staging.",
      environmentName: "Staging",
      evidence: [
        {
          type: "message",
          message: "Provider returned a safe error: rate_limited (fake).",
        },
      ],
      remediation: {
        type: "resync",
        label: "Retry sync after cooldown",
      },
      lastDetectedAt: RECENT,
    },
    {
      idSuffix: "verification-failed",
      ruleId: CORE_FINDING_RULE_IDS.CHANGE_VERIFICATION_FAILED,
      source: { type: "rayvan" },
      category: "configuration",
      severity: "error",
      title: "Verification failed after apply",
      summary: "Post-apply verification did not confirm expected values.",
      environmentName: "Staging",
      evidence: [
        {
          type: "message",
          message: "Observed fingerprint did not match desired fingerprint after apply.",
        },
      ],
      remediation: {
        type: "manual",
        label: "Inspect verification",
        instructions:
          "Compare desired vs observed values in the environment configuration editor, then re-run verification.",
      },
    },
    // Acknowledged + dismissed samples for filter demos
    {
      idSuffix: "acknowledged-info",
      ruleId: CORE_FINDING_RULE_IDS.ENVIRONMENT_NO_RESOURCES,
      source: { type: "rayvan" },
      category: "environment",
      severity: "info",
      title: "Local Scratch has no bound resources",
      summary: "Acknowledged: expected for workstation-only environments.",
      status: "acknowledged",
      environmentName: "Local Scratch",
      evidence: [{ type: "message", message: "No active resource bindings." }],
      firstDetectedAt: EARLIER,
      lastDetectedAt: EARLIER,
    },
    {
      idSuffix: "dismissed-noise",
      ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_UNMANAGED,
      source: { type: "rayvan" },
      category: "configuration",
      severity: "info",
      title: "Unmanaged key: VERCEL_URL",
      summary: "Dismissed as expected provider-managed value.",
      status: "dismissed",
      environmentName: "Preview",
      dismissalReason: "Expected behaviour",
      evidence: [{ type: "message", message: "Key exists remotely without local desired value." }],
    },
  ];

  return specs.map((spec) => buildRecord(project, spec, context));
}

function pluginFixtures(project: string, context?: FindingsSeedContext): FindingRecord[] {
  const specs: FixtureSpec[] = [
    {
      idSuffix: "plugin-vercel-deploy",
      ruleId: "vercel.deployment.failed",
      source: {
        type: "plugin",
        pluginId: "vercel",
        pluginVersion: "0.0.0-mock",
        connectionId: connId(project, "vercel", context?.connectionIdsByPluginId),
      },
      category: "deployment",
      severity: "critical",
      title: "[Mock plugin] Vercel deployment failed",
      summary: "Mock detection: latest Production deployment failed.",
      description:
        "DEVELOPMENT FIXTURE — plugin evaluate_findings capability simulation. Not a real Vercel API result.",
      environmentName: "Production",
      pluginId: "vercel",
      evidence: [
        {
          type: "deployment_state",
          deploymentId: `${project}:deploy:mock-1`,
          status: "failed",
          observedAt: RECENT,
        },
      ],
      remediation: {
        type: "open_integration",
        connectionId: connId(
          project,
          "vercel",
          context?.connectionIdsByPluginId,
        ),
        label: "Open Vercel integration",
      },
      metadata: { mockPluginFinding: true, pluginRule: "vercel.deployment.failed" },
      lastDetectedAt: RECENT,
    },
    {
      idSuffix: "plugin-supabase-rls",
      ruleId: "supabase.security.rls-disabled",
      source: {
        type: "plugin",
        pluginId: "supabase",
        pluginVersion: "0.0.0-mock",
        connectionId: connId(
          project,
          "supabase",
          context?.connectionIdsByPluginId,
        ),
      },
      category: "security",
      severity: "critical",
      title: "[Mock plugin] RLS disabled on public table",
      summary: "Mock detection: row-level security appears disabled.",
      description:
        "DEVELOPMENT FIXTURE — plugin evaluate_findings capability simulation.",
      environmentName: "Production",
      pluginId: "supabase",
      evidence: [
        {
          type: "message",
          message: "Table public.profiles reported rls_enabled=false (fake).",
        },
      ],
      remediation: {
        type: "manual",
        label: "Enable RLS",
        instructions:
          "In the Supabase dashboard, enable row-level security on the reported table and add policies.",
      },
      metadata: {
        mockPluginFinding: true,
        pluginRule: "supabase.security.rls-disabled",
      },
    },
    {
      idSuffix: "plugin-github-check",
      ruleId: "github.workflow.required-check-failed",
      source: {
        type: "plugin",
        pluginId: "github",
        pluginVersion: "0.0.0-mock",
        connectionId: connId(project, "github", context?.connectionIdsByPluginId),
      },
      category: "integration",
      severity: "error",
      title: "[Mock plugin] Required GitHub check failed",
      summary: "Mock detection: required status check did not pass.",
      description:
        "DEVELOPMENT FIXTURE — plugin evaluate_findings capability simulation.",
      pluginId: "github",
      evidence: [
        {
          type: "message",
          message: "Check “rayvan-ci” failed on main (fake).",
        },
      ],
      remediation: {
        type: "open_integration",
        connectionId: connId(
          project,
          "github",
          context?.connectionIdsByPluginId,
        ),
        label: "Open GitHub integration",
      },
      metadata: {
        mockPluginFinding: true,
        pluginRule: "github.workflow.required-check-failed",
      },
    },
    {
      idSuffix: "plugin-sentry-issue",
      ruleId: "sentry.issue.unresolved-spike",
      source: {
        type: "plugin",
        pluginId: "sentry",
        pluginVersion: "0.0.0-mock",
        connectionId: connId(project, "sentry", context?.connectionIdsByPluginId),
      },
      category: "availability",
      severity: "warning",
      title: "[Mock plugin] Sentry issue spike",
      summary: "Mock detection: unresolved issue volume spiked.",
      description:
        "DEVELOPMENT FIXTURE — plugin evaluate_findings capability simulation.",
      environmentName: "Production",
      pluginId: "sentry",
      evidence: [
        {
          type: "message",
          message: "Issue RAYVAN-42 increased 4× in the last hour (fake).",
        },
      ],
      remediation: {
        type: "open_integration",
        connectionId: connId(
          project,
          "sentry",
          context?.connectionIdsByPluginId,
        ),
        label: "Open Sentry integration",
      },
      metadata: {
        mockPluginFinding: true,
        pluginRule: "sentry.issue.unresolved-spike",
      },
    },
  ];

  return specs.map((spec) => buildRecord(project, spec, context));
}

export function buildDevFindingRecords(
  project: string,
  context?: FindingsSeedContext,
): FindingRecord[] {
  return [...coreFixtures(project, context), ...pluginFixtures(project, context)];
}

export function buildDevLifecycleEvents(
  records: FindingRecord[],
): FindingLifecycleEventRecord[] {
  const events: FindingLifecycleEventRecord[] = [];
  for (const record of records) {
    events.push({
      id: findingLifecycleEventId(`${record.id}:created`),
      findingId: record.id,
      projectId: record.projectId,
      type: "created",
      actor: DEV_FINDINGS_ENGINE_ACTOR,
      createdAt: record.firstDetectedAt,
      nextStatus: "open",
      metadata: {},
    });
    if (record.status === "acknowledged") {
      events.push({
        id: findingLifecycleEventId(`${record.id}:acknowledged`),
        findingId: record.id,
        projectId: record.projectId,
        type: "acknowledged",
        actor: DEV_FINDINGS_SYSTEM_ACTOR,
        createdAt: record.lastDetectedAt,
        previousStatus: "open",
        nextStatus: "acknowledged",
        reason: "Acknowledged in development fixtures",
        metadata: {},
      });
    }
    if (record.status === "dismissed") {
      events.push({
        id: findingLifecycleEventId(`${record.id}:dismissed`),
        findingId: record.id,
        projectId: record.projectId,
        type: "dismissed",
        actor: DEV_FINDINGS_SYSTEM_ACTOR,
        createdAt: record.lastDetectedAt,
        previousStatus: "open",
        nextStatus: "dismissed",
        reason: record.dismissalReason,
        metadata: {},
      });
    }
  }
  return events;
}

/** Minimal project context so evaluateProject can run core evaluators without crashing. */
export function buildDevFindingsProjectContext(
  project: string,
  context?: FindingsSeedContext,
): ProjectFindingsContext {
  const productionId = String(
    envId(project, "Production", context?.environmentIdsByName),
  );
  const stagingId = String(envId(project, "Staging", context?.environmentIdsByName));
  const developmentId = String(
    envId(project, "Development", context?.environmentIdsByName),
  );

  return {
    projectId: project,
    environments: [
      {
        id: environmentId(developmentId),
        projectId: projectId(project),
        name: "Development",
        slug: "development",
        kind: "development",
        status: "healthy",
        createdAt: NOW,
        updatedAt: NOW,
      },
      {
        id: environmentId(stagingId),
        projectId: projectId(project),
        name: "Staging",
        slug: "staging",
        kind: "staging",
        status: "healthy",
        createdAt: NOW,
        updatedAt: NOW,
      },
      {
        id: environmentId(productionId),
        projectId: projectId(project),
        name: "Production",
        slug: "production",
        kind: "production",
        status: "attention_required",
        createdAt: NOW,
        updatedAt: NOW,
      },
    ],
    keys: [],
    occurrences: [],
    desired: [],
    applied: [],
    connections: [
      {
        id: connId(project, "github", context?.connectionIdsByPluginId),
        pluginId: "github",
        projectId: project,
        name: "GitHub",
        status: "expired",
      },
      {
        id: connId(project, "vercel", context?.connectionIdsByPluginId),
        pluginId: "vercel",
        projectId: project,
        name: "Vercel",
        status: "connected",
      },
    ],
    installedPlugins: [
      {
        id: `${project}:plugin:github`,
        pluginId: "github",
        pluginVersion: "0.0.0-mock",
        status: "installed",
        enabled: true,
      },
      {
        id: `${project}:plugin:vercel`,
        pluginId: "vercel",
        pluginVersion: "0.0.0-mock",
        status: "installed",
        enabled: true,
      },
    ],
    discoveredResources: [
      {
        id: `${project}:discovered:unmapped`,
        pluginId: "vercel",
        connectionId: connId(project, "vercel", context?.connectionIdsByPluginId),
        name: "Unmapped preview env",
        resourceType: "environment",
        discoveryStatus: "active",
      },
    ],
    resourceBindings: [],
    mappingSuggestions: [
      {
        id: `${project}:suggestion:1`,
        projectId: project,
        connectionId: connId(project, "vercel", context?.connectionIdsByPluginId),
        discoveredResourceId: `${project}:discovered:unmapped`,
        suggestedEnvironmentName: "Preview",
        status: "pending",
      },
    ],
  };
}

export async function seedDevFindings(
  persistence: InMemoryFindingsPersistence,
  project: string,
  context?: FindingsSeedContext,
): Promise<void> {
  const records = buildDevFindingRecords(project, context);
  await persistence.findings.saveMany?.(records);
  if (!persistence.findings.saveMany) {
    for (const record of records) {
      await persistence.findings.save(record);
    }
  }
  for (const event of buildDevLifecycleEvents(records)) {
    await persistence.lifecycleEvents.append(event);
  }
}
