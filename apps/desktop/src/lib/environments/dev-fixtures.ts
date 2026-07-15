import type {
  ConfigurationValueAccess,
  ConfigurationValueType,
  Environment,
  EnvironmentKind,
  EnvironmentStatus,
} from "@rayvan/core";
import type {
  ConfigurationDesiredStateService,
  ConfigurationService,
  DiscoveredResourceRecord,
  DiscoveredResourceRepository,
  EnvironmentMappingService,
  EnvironmentService,
  InstalledPluginRecord,
  InstalledPluginRepository,
  PluginConnectionRecord,
  PluginConnectionRepository,
  PluginPermissionGrantRepository,
  ResourceBindingService,
} from "@rayvan/local-database";

import {
  DEV_FIXTURE_SYSTEM_ACTOR,
  ensureCatalogInstalled,
  seedProjectConnections,
} from "../plugins/dev-fixtures.js";

/**
 * DEVELOPMENT ONLY seed data for the Environments workspace.
 * Fake values only — never real secrets or provider credentials.
 */

export const ENVIRONMENTS_FIXTURE_ACTOR = DEV_FIXTURE_SYSTEM_ACTOR;

interface EnvSeedSpec {
  name: string;
  kind: EnvironmentKind;
  status: EnvironmentStatus;
  description?: string;
}

const ENVIRONMENT_SPECS: readonly EnvSeedSpec[] = [
  {
    name: "Development",
    kind: "development",
    status: "healthy",
    description: "Local and shared development targets.",
  },
  {
    name: "Staging",
    kind: "staging",
    status: "healthy",
    description: "Pre-production validation environment.",
  },
  {
    name: "Production",
    kind: "production",
    status: "attention_required",
    description: "Live customer-facing environment.",
  },
  {
    name: "Preview",
    kind: "preview",
    status: "healthy",
    description: "Ephemeral preview deployments.",
  },
  {
    name: "Local Scratch",
    kind: "custom",
    status: "local_only",
    description: "Workstation-only custom environment (no sync).",
  },
];

interface ResourceSeedSpec {
  pluginId: string;
  providerResourceId: string;
  resourceType: string;
  name: string;
  bindToEnv?: string;
  unmapped?: boolean;
  suggestToEnv?: string;
  suggestionReasons?: string[];
  suggestionConfidence?: number;
}

const RESOURCE_SPECS: readonly ResourceSeedSpec[] = [
  {
    pluginId: "vercel",
    providerResourceId: "env:development",
    resourceType: "environment",
    name: "Vercel Development",
    bindToEnv: "Development",
  },
  {
    pluginId: "vercel",
    providerResourceId: "env:preview",
    resourceType: "environment",
    name: "Vercel Preview",
    bindToEnv: "Preview",
  },
  {
    pluginId: "vercel",
    providerResourceId: "env:staging",
    resourceType: "environment",
    name: "Vercel Staging",
    bindToEnv: "Staging",
  },
  {
    pluginId: "vercel",
    providerResourceId: "env:production",
    resourceType: "environment",
    name: "Vercel Production",
    bindToEnv: "Production",
  },
  {
    pluginId: "supabase",
    providerResourceId: "project:dev",
    resourceType: "project",
    name: "Supabase Dev",
    bindToEnv: "Development",
  },
  {
    pluginId: "supabase",
    providerResourceId: "project:staging",
    resourceType: "project",
    name: "Supabase Staging",
    bindToEnv: "Staging",
  },
  {
    pluginId: "supabase",
    providerResourceId: "project:prod",
    resourceType: "project",
    name: "Supabase Production",
    bindToEnv: "Production",
  },
  {
    pluginId: "github",
    providerResourceId: "branch:main",
    resourceType: "branch",
    name: "GitHub main",
    bindToEnv: "Production",
  },
  {
    pluginId: "github",
    providerResourceId: "branch:develop",
    resourceType: "branch",
    name: "GitHub develop",
    unmapped: true,
    suggestToEnv: "Development",
    suggestionReasons: ["Branch name matches Development", "High confidence name match"],
    suggestionConfidence: 0.92,
  },
  {
    pluginId: "sentry",
    providerResourceId: "env:production",
    resourceType: "environment",
    name: "Sentry Production",
    bindToEnv: "Production",
  },
  {
    pluginId: "sentry",
    providerResourceId: "env:staging",
    resourceType: "environment",
    name: "Sentry Staging",
    unmapped: true,
    suggestToEnv: "Staging",
    suggestionReasons: ["Provider environment label matches Staging"],
    suggestionConfidence: 0.88,
  },
  {
    pluginId: "example-local",
    providerResourceId: "file:.env",
    resourceType: "env_file",
    name: "Local .env",
    bindToEnv: "Development",
  },
  {
    pluginId: "example-local",
    providerResourceId: "file:.env.test",
    resourceType: "env_file",
    name: "Local .env.test",
    unmapped: true,
    suggestToEnv: "Development",
    suggestionReasons: ["Local env file often maps to Development"],
    suggestionConfidence: 0.75,
  },
];

interface KeySeedSpec {
  name: string;
  valueType: ConfigurationValueType;
  required?: boolean;
  sensitive?: boolean;
  description?: string;
}

const KEY_SPECS: readonly KeySeedSpec[] = [
  {
    name: "API_BASE_URL",
    valueType: "url",
    required: true,
    description: "Public API base URL",
  },
  {
    name: "DATABASE_URL",
    valueType: "secret",
    required: true,
    sensitive: true,
    description: "Primary database connection string",
  },
  {
    name: "SUPABASE_URL",
    valueType: "url",
    required: true,
  },
  {
    name: "SUPABASE_ANON_KEY",
    valueType: "secret",
    required: true,
    sensitive: true,
  },
  {
    name: "SENTRY_DSN",
    valueType: "secret",
    required: false,
    sensitive: true,
  },
  {
    name: "STRIPE_SECRET_KEY",
    valueType: "secret",
    required: true,
    sensitive: true,
  },
  {
    name: "DEBUG_MODE",
    valueType: "boolean",
    required: false,
  },
  {
    name: "NODE_ENV",
    valueType: "string",
    required: true,
  },
  {
    name: "VERCEL_URL",
    valueType: "url",
    required: false,
  },
  {
    name: "GITHUB_TOKEN",
    valueType: "secret",
    required: false,
    sensitive: true,
  },
];

interface OccurrenceSeedSpec {
  keyName: string;
  envName: string;
  pluginId: string;
  resourceProviderId: string;
  valueAccess: ConfigurationValueAccess;
  observedValue?: string;
  maskedValue?: string;
  valueFingerprint?: string;
  secretValueRef?: string;
  /** Override lastObservedAt for stale scenarios. */
  lastObservedAt?: string;
}

/** Deterministic fake values only. */
const OCCURRENCE_SPECS: readonly OccurrenceSeedSpec[] = [
  // Matching API_BASE_URL in Dev + Staging (same fingerprint/value)
  {
    keyName: "API_BASE_URL",
    envName: "Development",
    pluginId: "vercel",
    resourceProviderId: "env:development",
    valueAccess: "readable",
    observedValue: "https://api.example.rayvan.dev",
    valueFingerprint: "fp:dev-api",
  },
  {
    keyName: "API_BASE_URL",
    envName: "Staging",
    pluginId: "vercel",
    resourceProviderId: "env:staging",
    valueAccess: "readable",
    observedValue: "https://api.example.rayvan.dev",
    valueFingerprint: "fp:dev-api",
  },
  // Mismatch in Production
  {
    keyName: "API_BASE_URL",
    envName: "Production",
    pluginId: "vercel",
    resourceProviderId: "env:production",
    valueAccess: "readable",
    observedValue: "https://api.example.rayvan.com",
    valueFingerprint: "fp:prod-api",
  },
  // Multi-plugin same key in Development
  {
    keyName: "API_BASE_URL",
    envName: "Development",
    pluginId: "example-local",
    resourceProviderId: "file:.env",
    valueAccess: "readable",
    observedValue: "https://api.example.rayvan.dev",
    valueFingerprint: "fp:dev-api",
  },
  // Sensitive masked DATABASE_URL
  {
    keyName: "DATABASE_URL",
    envName: "Development",
    pluginId: "supabase",
    resourceProviderId: "project:dev",
    valueAccess: "masked",
    maskedValue: "postgres://••••@db.example.rayvan.dev:5432/rayvan",
    valueFingerprint: "fp:db-dev",
    secretValueRef: "cred:db-dev",
  },
  {
    keyName: "DATABASE_URL",
    envName: "Staging",
    pluginId: "supabase",
    resourceProviderId: "project:staging",
    valueAccess: "masked",
    maskedValue: "postgres://••••@db.staging.example.rayvan.dev:5432/rayvan",
    valueFingerprint: "fp:db-staging",
    secretValueRef: "cred:db-staging",
  },
  {
    keyName: "DATABASE_URL",
    envName: "Production",
    pluginId: "supabase",
    resourceProviderId: "project:prod",
    valueAccess: "masked",
    maskedValue: "postgres://••••@db.prod.example.rayvan.dev:5432/rayvan",
    valueFingerprint: "fp:db-prod",
    secretValueRef: "cred:db-prod",
  },
  // Matching SUPABASE_URL
  {
    keyName: "SUPABASE_URL",
    envName: "Development",
    pluginId: "supabase",
    resourceProviderId: "project:dev",
    valueAccess: "readable",
    observedValue: "https://dev.supabase.example.rayvan.dev",
    valueFingerprint: "fp:sb-url-dev",
  },
  {
    keyName: "SUPABASE_URL",
    envName: "Staging",
    pluginId: "supabase",
    resourceProviderId: "project:staging",
    valueAccess: "readable",
    observedValue: "https://staging.supabase.example.rayvan.dev",
    valueFingerprint: "fp:sb-url-staging",
  },
  {
    keyName: "SUPABASE_URL",
    envName: "Production",
    pluginId: "supabase",
    resourceProviderId: "project:prod",
    valueAccess: "readable",
    observedValue: "https://prod.supabase.example.rayvan.dev",
    valueFingerprint: "fp:sb-url-prod",
  },
  {
    keyName: "SUPABASE_ANON_KEY",
    envName: "Development",
    pluginId: "supabase",
    resourceProviderId: "project:dev",
    valueAccess: "masked",
    maskedValue: "eyJ••••anon",
    valueFingerprint: "fp:sb-anon-dev",
    secretValueRef: "cred:sb-anon-dev",
  },
  {
    keyName: "SUPABASE_ANON_KEY",
    envName: "Production",
    pluginId: "supabase",
    resourceProviderId: "project:prod",
    valueAccess: "masked",
    maskedValue: "eyJ••••anon",
    valueFingerprint: "fp:sb-anon-prod",
    secretValueRef: "cred:sb-anon-prod",
  },
  // Missing required STRIPE in Staging (only Production + Dev)
  {
    keyName: "STRIPE_SECRET_KEY",
    envName: "Development",
    pluginId: "vercel",
    resourceProviderId: "env:development",
    valueAccess: "locked",
    maskedValue: "sk_test_fake_••••",
    valueFingerprint: "fp:stripe-dev",
    secretValueRef: "cred:stripe-dev",
  },
  {
    keyName: "STRIPE_SECRET_KEY",
    envName: "Production",
    pluginId: "vercel",
    resourceProviderId: "env:production",
    valueAccess: "locked",
    maskedValue: "sk_live_fake_••••",
    valueFingerprint: "fp:stripe-prod",
    secretValueRef: "cred:stripe-prod",
  },
  {
    keyName: "SENTRY_DSN",
    envName: "Production",
    pluginId: "sentry",
    resourceProviderId: "env:production",
    valueAccess: "masked",
    maskedValue: "https://••••@o0.ingest.sentry.io/0",
    valueFingerprint: "fp:sentry-prod",
    secretValueRef: "cred:sentry-prod",
  },
  {
    keyName: "DEBUG_MODE",
    envName: "Development",
    pluginId: "example-local",
    resourceProviderId: "file:.env",
    valueAccess: "readable",
    observedValue: "true",
    valueFingerprint: "fp:debug-true",
  },
  {
    keyName: "DEBUG_MODE",
    envName: "Production",
    pluginId: "vercel",
    resourceProviderId: "env:production",
    valueAccess: "readable",
    observedValue: "false",
    valueFingerprint: "fp:debug-false",
  },
  {
    keyName: "NODE_ENV",
    envName: "Development",
    pluginId: "vercel",
    resourceProviderId: "env:development",
    valueAccess: "readable",
    observedValue: "development",
    valueFingerprint: "fp:node-dev",
  },
  {
    keyName: "NODE_ENV",
    envName: "Staging",
    pluginId: "vercel",
    resourceProviderId: "env:staging",
    valueAccess: "readable",
    observedValue: "staging",
    valueFingerprint: "fp:node-staging",
    lastObservedAt: "2026-01-01T00:00:00.000Z",
  },
  {
    keyName: "NODE_ENV",
    envName: "Production",
    pluginId: "vercel",
    resourceProviderId: "env:production",
    valueAccess: "readable",
    observedValue: "production",
    valueFingerprint: "fp:node-prod",
  },
  // Preview-only key (env-unique)
  {
    keyName: "VERCEL_URL",
    envName: "Preview",
    pluginId: "vercel",
    resourceProviderId: "env:preview",
    valueAccess: "readable",
    observedValue: "https://preview.example.rayvan.dev",
    valueFingerprint: "fp:vercel-url-preview",
  },
  // name_only — no value comparison
  {
    keyName: "GITHUB_TOKEN",
    envName: "Production",
    pluginId: "github",
    resourceProviderId: "branch:main",
    valueAccess: "name_only",
  },
  // Missing-local / not_managed: discovered in Development without desired
  {
    keyName: "VERCEL_URL",
    envName: "Development",
    pluginId: "vercel",
    resourceProviderId: "env:development",
    valueAccess: "readable",
    observedValue: "https://dev.example.rayvan.dev",
    valueFingerprint: "fp:vercel-url-dev-only",
  },
];

export interface SeedEnvironmentsDeps {
  installedPlugins: InstalledPluginRepository;
  connections: PluginConnectionRepository;
  permissionGrants: PluginPermissionGrantRepository;
  discoveredResources: DiscoveredResourceRepository;
  environmentService: EnvironmentService;
  configurationService: ConfigurationService;
  desiredStateService: ConfigurationDesiredStateService;
  bindingService: ResourceBindingService;
  mappingService: EnvironmentMappingService;
}

export interface SeedEnvironmentsResult {
  environmentsByName: Map<string, Environment>;
  connectionsByPluginId: Map<string, PluginConnectionRecord>;
  resourcesByProviderKey: Map<string, DiscoveredResourceRecord>;
}

/**
 * Seeds catalog connections (via plugin fixtures), environments, resources,
 * bindings, configuration, and pending mapping suggestions for a project.
 * Callers must ensure this runs once per project id.
 */
export async function seedProjectEnvironments(
  deps: SeedEnvironmentsDeps,
  projectId: string,
): Promise<SeedEnvironmentsResult> {
  const installedByPluginId = await ensureCatalogInstalled(deps.installedPlugins);
  await seedProjectConnections(
    {
      connections: deps.connections,
      permissionGrants: deps.permissionGrants,
    },
    projectId,
    installedByPluginId,
  );

  const connections = await deps.connections.listByProjectId(projectId);
  const connectionsByPluginId = new Map<string, PluginConnectionRecord>();
  for (const connection of connections) {
    connectionsByPluginId.set(connection.pluginId, connection);
  }

  const environmentsByName = new Map<string, Environment>();
  for (const spec of ENVIRONMENT_SPECS) {
    const environment = await deps.environmentService.create({
      projectId,
      name: spec.name,
      kind: spec.kind,
      description: spec.description,
      status: spec.status,
    });
    environmentsByName.set(spec.name, environment);
  }

  const resourcesByProviderKey = new Map<string, DiscoveredResourceRecord>();
  const now = new Date().toISOString();

  for (const spec of RESOURCE_SPECS) {
    const connection = connectionsByPluginId.get(spec.pluginId);
    const installed = installedByPluginId.get(spec.pluginId);
    if (!connection || !installed) {
      continue;
    }

    const resource: DiscoveredResourceRecord = {
      id: crypto.randomUUID(),
      pluginId: spec.pluginId,
      installedPluginId: installed.id,
      connectionId: connection.id,
      providerResourceId: spec.providerResourceId,
      resourceType: spec.resourceType,
      name: spec.name,
      metadata: { fixture: true },
      pluginVersion: installed.pluginVersion,
      schemaVersion: "1",
      discoveryStatus: "active",
      firstDiscoveredAt: now,
      lastDiscoveredAt: now,
    };
    await deps.discoveredResources.save(resource);
    resourcesByProviderKey.set(
      `${spec.pluginId}:${spec.providerResourceId}`,
      resource,
    );

    if (spec.bindToEnv) {
      const environment = environmentsByName.get(spec.bindToEnv);
      if (environment) {
        await deps.bindingService.bind({
          projectId,
          environmentId: environment.id,
          expectedProjectIdForEnvironment: projectId,
          discoveredResourceId: resource.id,
          displayName: spec.name,
          createdBy: ENVIRONMENTS_FIXTURE_ACTOR,
        });
      }
    }

    if (spec.unmapped && spec.suggestToEnv) {
      const suggested = environmentsByName.get(spec.suggestToEnv);
      await deps.mappingService.createSuggestion({
        projectId,
        connectionId: connection.id,
        discoveredResourceId: resource.id,
        suggestedEnvironmentId: suggested?.id,
        suggestedEnvironmentName: spec.suggestToEnv,
        confidence: spec.suggestionConfidence ?? 0.8,
        reasons: spec.suggestionReasons ?? ["Fixture mapping suggestion"],
      });
    }
  }

  const keysByName = new Map<string, { id: string }>();
  for (const keySpec of KEY_SPECS) {
    const key = await deps.configurationService.upsertKeyByName(
      projectId,
      keySpec.name,
      {
        valueType: keySpec.valueType,
        required: keySpec.required,
        sensitive: keySpec.sensitive ?? keySpec.valueType === "secret",
        description: keySpec.description,
        source: "discovered",
      },
    );
    keysByName.set(key.name, key);
  }

  for (const occurrence of OCCURRENCE_SPECS) {
    const key = keysByName.get(occurrence.keyName);
    const environment = environmentsByName.get(occurrence.envName);
    const resource = resourcesByProviderKey.get(
      `${occurrence.pluginId}:${occurrence.resourceProviderId}`,
    );
    const connection = connectionsByPluginId.get(occurrence.pluginId);
    if (!key || !environment || !resource || !connection) {
      continue;
    }

    const bindings = await deps.bindingService.listByProjectId(projectId);
    const binding = bindings.find(
      (item) =>
        item.discoveredResourceId === resource.id &&
        item.environmentId === environment.id &&
        item.bindingStatus === "active",
    );

    const upserted = await deps.configurationService.upsertOccurrence({
      configurationKeyId: key.id,
      projectId,
      environmentId: environment.id,
      pluginId: occurrence.pluginId,
      connectionId: connection.id,
      discoveredResourceId: resource.id,
      resourceBindingId: binding?.id,
      providerKey: occurrence.keyName,
      valueAccess: occurrence.valueAccess,
      observedValue: occurrence.observedValue,
      maskedValue: occurrence.maskedValue,
      valueFingerprint: occurrence.valueFingerprint,
      secretValueRef: occurrence.secretValueRef,
    });

    if (occurrence.lastObservedAt) {
      // Force stale lastObservedAt after upsert (service sets "now").
      await deps.configurationService.upsertOccurrence({
        configurationKeyId: key.id,
        projectId,
        environmentId: environment.id,
        pluginId: occurrence.pluginId,
        connectionId: connection.id,
        discoveredResourceId: resource.id,
        resourceBindingId: binding?.id,
        providerKey: occurrence.keyName,
        valueAccess: occurrence.valueAccess,
        observedValue: occurrence.observedValue,
        maskedValue: occurrence.maskedValue,
        valueFingerprint: occurrence.valueFingerprint,
        secretValueRef: occurrence.secretValueRef,
      });
      void upserted;
      // Memory repo update path sets lastObservedAt to now — patch via occurrence list.
      const listed = await deps.configurationService.listOccurrencesByEnvironment(
        environment.id,
      );
      const match = listed.find((item) => item.id === upserted.id);
      if (match) {
        (match as { lastObservedAt: string }).lastObservedAt =
          occurrence.lastObservedAt;
      }
    }
  }

  const actor = {
    kind: "system" as const,
    id: ENVIRONMENTS_FIXTURE_ACTOR.id,
  };
  const bindings = await deps.bindingService.listByProjectId(projectId);

  const findBinding = (envName: string, pluginId: string, providerId: string) => {
    const environment = environmentsByName.get(envName);
    const resource = resourcesByProviderKey.get(`${pluginId}:${providerId}`);
    if (!environment || !resource) {
      return undefined;
    }
    return bindings.find(
      (item) =>
        item.discoveredResourceId === resource.id &&
        item.environmentId === environment.id &&
        item.bindingStatus === "active",
    );
  };

  const keyId = (name: string) => keysByName.get(name)?.id;

  // --- Desired + applied fixture scenarios (fake values only) ---

  // Staging: mostly in sync (desired matches observed + applied recorded)
  const staging = environmentsByName.get("Staging");
  const stagingApiKey = keyId("API_BASE_URL");
  const stagingNodeKey = keyId("NODE_ENV");
  const stagingBinding = findBinding("Staging", "vercel", "env:staging");
  if (staging && stagingApiKey && stagingBinding) {
    const desiredApi = await deps.desiredStateService.saveDesired({
      configurationKeyId: stagingApiKey,
      environmentId: staging.id,
      projectId,
      desiredValue: "https://api.example.rayvan.dev",
      valueFingerprint: "fp:dev-api",
      updatedBy: actor,
    });
    await deps.desiredStateService.recordApplied({
      configurationKeyId: stagingApiKey,
      environmentId: staging.id,
      projectId,
      resourceBindingId: stagingBinding.id,
      desiredRevision: desiredApi.revision,
      appliedFingerprint: "fp:dev-api",
      applyExecutionId: "fixture-apply-staging-api",
      status: "verified",
      verifiedAt: new Date().toISOString(),
    });
  }
  if (staging && stagingNodeKey && stagingBinding) {
    const desiredNode = await deps.desiredStateService.saveDesired({
      configurationKeyId: stagingNodeKey,
      environmentId: staging.id,
      projectId,
      desiredValue: "staging",
      valueFingerprint: "fp:node-staging",
      updatedBy: actor,
    });
    await deps.desiredStateService.recordApplied({
      configurationKeyId: stagingNodeKey,
      environmentId: staging.id,
      projectId,
      resourceBindingId: stagingBinding.id,
      desiredRevision: desiredNode.revision,
      appliedFingerprint: "fp:node-staging",
      applyExecutionId: "fixture-apply-staging-node",
      status: "applied",
    });
  }

  // Production: saved changes not applied (desired ≠ observed)
  const production = environmentsByName.get("Production");
  const prodApiKey = keyId("API_BASE_URL");
  const prodBinding = findBinding("Production", "vercel", "env:production");
  if (production && prodApiKey) {
    await deps.desiredStateService.saveDesired({
      configurationKeyId: prodApiKey,
      environmentId: production.id,
      projectId,
      desiredValue: "https://api.example.rayvan.com/v2",
      valueFingerprint: "fp:prod-api-v2",
      updatedBy: actor,
    });
  }

  // Production: remote changed outside Rayvan
  // (applied matches old desired, observed differs, desired unchanged)
  const prodDebugKey = keyId("DEBUG_MODE");
  if (production && prodDebugKey && prodBinding) {
    const desiredDebug = await deps.desiredStateService.saveDesired({
      configurationKeyId: prodDebugKey,
      environmentId: production.id,
      projectId,
      desiredValue: "false",
      valueFingerprint: "fp:debug-false",
      updatedBy: actor,
    });
    await deps.desiredStateService.recordApplied({
      configurationKeyId: prodDebugKey,
      environmentId: production.id,
      projectId,
      resourceBindingId: prodBinding.id,
      desiredRevision: desiredDebug.revision,
      appliedFingerprint: "fp:debug-false",
      applyExecutionId: "fixture-apply-prod-debug",
      status: "applied",
    });
    // Observed was seeded as false — flip observed fingerprint/value to simulate remote change.
    const prodOcc = (
      await deps.configurationService.listOccurrencesByEnvironment(production.id)
    ).find(
      (item) =>
        item.configurationKeyId === prodDebugKey &&
        item.resourceBindingId === prodBinding.id,
    );
    if (prodOcc) {
      await deps.configurationService.upsertOccurrence({
        configurationKeyId: prodDebugKey,
        projectId,
        environmentId: production.id,
        pluginId: prodOcc.pluginId,
        connectionId: prodOcc.connectionId,
        discoveredResourceId: prodOcc.discoveredResourceId,
        resourceBindingId: prodOcc.resourceBindingId,
        providerKey: prodOcc.providerKey,
        valueAccess: "readable",
        observedValue: "true",
        valueFingerprint: "fp:debug-true-remote",
      });
    }
  }

  // Production: missing remote — desired STRIPE exists but Staging has no occurrence;
  // also seed desired for Staging STRIPE (missing remotely).
  const stripeKey = keyId("STRIPE_SECRET_KEY");
  if (staging && stripeKey) {
    await deps.desiredStateService.saveDesired({
      configurationKeyId: stripeKey,
      environmentId: staging.id,
      projectId,
      secretValueRef: "cred:stripe-staging-desired",
      valueFingerprint: "fp:stripe-staging-desired",
      updatedBy: actor,
    });
  }

  // Sensitive desired with secretValueRef (Development DATABASE_URL)
  const development = environmentsByName.get("Development");
  const dbKey = keyId("DATABASE_URL");
  const devSbBinding = findBinding("Development", "supabase", "project:dev");
  if (development && dbKey && devSbBinding) {
    const desiredDb = await deps.desiredStateService.saveDesired({
      configurationKeyId: dbKey,
      environmentId: development.id,
      projectId,
      secretValueRef: "cred:db-dev-desired",
      valueFingerprint: "fp:db-dev",
      updatedBy: actor,
    });
    await deps.desiredStateService.recordApplied({
      configurationKeyId: dbKey,
      environmentId: development.id,
      projectId,
      resourceBindingId: devSbBinding.id,
      desiredRevision: desiredDb.revision,
      appliedFingerprint: "fp:db-dev",
      applyExecutionId: "fixture-apply-dev-db",
      status: "verified",
      verifiedAt: new Date().toISOString(),
    });
  }

  // Partially applied: Development API_BASE_URL — vercel applied, local .env not
  const apiKey = keyId("API_BASE_URL");
  const devVercelBinding = findBinding("Development", "vercel", "env:development");
  const devLocalBinding = findBinding("Development", "example-local", "file:.env");
  if (development && apiKey && devVercelBinding) {
    const desiredApi = await deps.desiredStateService.saveDesired({
      configurationKeyId: apiKey,
      environmentId: development.id,
      projectId,
      desiredValue: "https://api.example.rayvan.dev",
      valueFingerprint: "fp:dev-api",
      updatedBy: actor,
    });
    await deps.desiredStateService.recordApplied({
      configurationKeyId: apiKey,
      environmentId: development.id,
      projectId,
      resourceBindingId: devVercelBinding.id,
      desiredRevision: desiredApi.revision,
      appliedFingerprint: "fp:dev-api",
      applyExecutionId: "fixture-apply-dev-api-vercel",
      status: "applied",
    });
    if (devLocalBinding) {
      await deps.desiredStateService.recordApplied({
        configurationKeyId: apiKey,
        environmentId: development.id,
        projectId,
        resourceBindingId: devLocalBinding.id,
        desiredRevision: desiredApi.revision,
        appliedFingerprint: "fp:dev-api-stale",
        applyExecutionId: "fixture-apply-dev-api-local-fail",
        status: "failed",
      });
    }
  }

  // Apply / verification failure rows (Production SENTRY)
  const sentryKey = keyId("SENTRY_DSN");
  const sentryBinding = findBinding("Production", "sentry", "env:production");
  if (production && sentryKey && sentryBinding) {
    const desiredSentry = await deps.desiredStateService.saveDesired({
      configurationKeyId: sentryKey,
      environmentId: production.id,
      projectId,
      secretValueRef: "cred:sentry-prod-desired",
      valueFingerprint: "fp:sentry-prod",
      updatedBy: actor,
    });
    await deps.desiredStateService.recordApplied({
      configurationKeyId: sentryKey,
      environmentId: production.id,
      projectId,
      resourceBindingId: sentryBinding.id,
      desiredRevision: desiredSentry.revision,
      appliedFingerprint: "fp:sentry-prod",
      applyExecutionId: "fixture-apply-sentry",
      verificationExecutionId: "fixture-verify-sentry",
      status: "verification_failed",
      verifiedAt: new Date().toISOString(),
    });
  }

  // Locked remote secret desired (Production GITHUB_TOKEN) — occurrence is name_only
  const ghKey = keyId("GITHUB_TOKEN");
  if (production && ghKey) {
    await deps.desiredStateService.saveDesired({
      configurationKeyId: ghKey,
      environmentId: production.id,
      projectId,
      secretValueRef: "cred:gh-prod-desired",
      valueFingerprint: "fp:gh-prod",
      updatedBy: actor,
    });
  }

  // Mark Production as having sync-error metadata in presentation (status already attention_required).
  // Do NOT encode config sync into Environment.status — cards use status service aggregates.
  if (production) {
    await deps.environmentService.update(production.id, {
      status: "attention_required",
      description:
        "Live customer-facing environment. Last sync reported attention required for Sentry.",
    });
  }

  return { environmentsByName, connectionsByPluginId, resourcesByProviderKey };
}

export async function ensureEnvironmentsCatalogInstalled(
  installedPlugins: InstalledPluginRepository,
): Promise<Map<string, InstalledPluginRecord>> {
  return ensureCatalogInstalled(installedPlugins);
}
