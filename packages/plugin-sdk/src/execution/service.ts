import { CAPABILITY_HANDLER_KEYS } from "../capabilities/index.js";
import type {
  ApplyResult,
  AuthenticateResult,
  ChangePlan,
  DiscoveredResource,
  ObservedResourceState,
  VerificationResult,
} from "../contracts/index.js";
import {
  PluginError,
  PluginExecutionError,
  PluginValidationError,
} from "../errors/index.js";
import type { PluginCapability, PluginPermission } from "../manifest/index.js";
import type { PluginRegistry } from "../registry/index.js";
import {
  validateApplyResult,
  validateAuthenticateResult,
  validateChangePlan,
  validateDiscoveredResource,
  validateObservedResourceState,
  validateVerificationResult,
} from "../validation/index.js";
import { assertApplyGuards } from "./apply-guards.js";
import type { PluginExecutionEventSink } from "./events/types.js";
import { DEFAULT_CAPABILITY_PERMISSIONS } from "./permissions/capability-permissions.js";
import type {
  PluginCapabilityPermissionPolicy,
  PluginPermissionResolver,
} from "./permissions/types.js";
import { redactSecrets } from "./redaction.js";
import type {
  ApplyExecutionRequest,
  AuthenticateExecutionRequest,
  DiscoverExecutionRequest,
  InspectExecutionRequest,
  PlanExecutionRequest,
  PluginExecutionRequestBase,
  VerifyExecutionRequest,
} from "./requests.js";
import type {
  PluginExecutionErrorCode,
  PluginExecutionResult,
  PluginExecutionStatus,
  SerializedPluginExecutionError,
} from "./results.js";
import type { PluginRuntime } from "./runtime/types.js";
import { DEFAULT_PLUGIN_TIMEOUTS } from "./timeouts.js";

export interface PluginExecutionServiceOptions {
  registry: PluginRegistry;
  runtime: PluginRuntime;
  permissionResolver: PluginPermissionResolver;
  eventSink?: PluginExecutionEventSink;
  capabilityPermissions?: PluginCapabilityPermissionPolicy;
  idFactory?: () => string;
  now?: () => Date;
}

interface ExecuteCapabilityOptions<TInput, TOutput> {
  request: PluginExecutionRequestBase;
  capability: PluginCapability;
  input: TInput;
  validateInput: () => void;
  validateOutput: (output: TOutput) => void;
  runApplyGuards?: () => void;
}

/**
 * Trusted execution boundary for all plugin capability invocations.
 * Callers must use this service rather than calling plugin handlers directly.
 */
export interface IPluginExecutionService {
  authenticate(
    request: AuthenticateExecutionRequest,
  ): Promise<PluginExecutionResult<AuthenticateResult>>;
  discover(
    request: DiscoverExecutionRequest,
  ): Promise<PluginExecutionResult<DiscoveredResource[]>>;
  inspect(
    request: InspectExecutionRequest,
  ): Promise<PluginExecutionResult<ObservedResourceState>>;
  plan(
    request: PlanExecutionRequest,
  ): Promise<PluginExecutionResult<ChangePlan>>;
  apply(
    request: ApplyExecutionRequest,
  ): Promise<PluginExecutionResult<ApplyResult>>;
  verify(
    request: VerifyExecutionRequest,
  ): Promise<PluginExecutionResult<VerificationResult>>;
}

export class PluginExecutionService implements IPluginExecutionService {
  private readonly registry: PluginRegistry;
  private readonly runtime: PluginRuntime;
  private readonly permissionResolver: PluginPermissionResolver;
  private readonly eventSink: PluginExecutionEventSink | undefined;
  private readonly capabilityPermissions: PluginCapabilityPermissionPolicy;
  private readonly idFactory: () => string;
  private readonly now: () => Date;

  constructor(options: PluginExecutionServiceOptions) {
    this.registry = options.registry;
    this.runtime = options.runtime;
    this.permissionResolver = options.permissionResolver;
    this.eventSink = options.eventSink;
    this.capabilityPermissions =
      options.capabilityPermissions ?? DEFAULT_CAPABILITY_PERMISSIONS;
    this.idFactory = options.idFactory ?? (() => crypto.randomUUID());
    this.now = options.now ?? (() => new Date());
  }

  authenticate(
    request: AuthenticateExecutionRequest,
  ): Promise<PluginExecutionResult<AuthenticateResult>> {
    return this.executeCapability({
      request,
      capability: "authenticate",
      input: request.context,
      validateInput: () => {
        this.assertRequestBase(request);
        this.assertMatchingPluginId(request.pluginId, request.context.pluginId);
      },
      validateOutput: (output) =>
        validateAuthenticateResult(output, request.pluginId),
    });
  }

  discover(
    request: DiscoverExecutionRequest,
  ): Promise<PluginExecutionResult<DiscoveredResource[]>> {
    return this.executeCapability({
      request,
      capability: "discover",
      input: request.context,
      validateInput: () => {
        this.assertRequestBase(request);
        this.assertMatchingPluginId(request.pluginId, request.context.pluginId);
      },
      validateOutput: (output) => {
        if (!Array.isArray(output)) {
          throw new PluginValidationError(
            "discover result must be an array",
            { pluginId: request.pluginId },
          );
        }
        for (const resource of output) {
          validateDiscoveredResource(resource, request.pluginId);
        }
      },
    });
  }

  inspect(
    request: InspectExecutionRequest,
  ): Promise<PluginExecutionResult<ObservedResourceState>> {
    return this.executeCapability({
      request,
      capability: "inspect",
      input: request.context,
      validateInput: () => {
        this.assertRequestBase(request);
        this.assertMatchingPluginId(request.pluginId, request.context.pluginId);
        this.assertResourceBinding(request, request.context.resource.resourceId);
      },
      validateOutput: (output) => validateObservedResourceState(output),
    });
  }

  plan(
    request: PlanExecutionRequest,
  ): Promise<PluginExecutionResult<ChangePlan>> {
    return this.executeCapability({
      request,
      capability: "plan",
      input: request.context,
      validateInput: () => {
        this.assertRequestBase(request);
        this.assertMatchingPluginId(request.pluginId, request.context.pluginId);
        this.assertResourceBinding(request, request.context.resource.resourceId);
      },
      validateOutput: (output) => validateChangePlan(output),
    });
  }

  apply(
    request: ApplyExecutionRequest,
  ): Promise<PluginExecutionResult<ApplyResult>> {
    return this.executeCapability({
      request,
      capability: "apply",
      input: request.context,
      validateInput: () => {
        this.assertRequestBase(request);
        this.assertMatchingPluginId(request.pluginId, request.context.pluginId);
        this.assertResourceBinding(request, request.context.resource.resourceId);
        if (
          request.context.approvedPlan === null ||
          typeof request.context.approvedPlan !== "object"
        ) {
          throw new PluginValidationError(
            "apply context must include an ApprovedChangePlan",
            { pluginId: request.pluginId },
          );
        }
      },
      runApplyGuards: () =>
        assertApplyGuards(request.pluginId, request.context),
      validateOutput: (output) =>
        validateApplyResult(output, request.pluginId),
    });
  }

  verify(
    request: VerifyExecutionRequest,
  ): Promise<PluginExecutionResult<VerificationResult>> {
    return this.executeCapability({
      request,
      capability: "verify",
      input: request.context,
      validateInput: () => {
        this.assertRequestBase(request);
        this.assertMatchingPluginId(request.pluginId, request.context.pluginId);
        this.assertResourceBinding(request, request.context.resource.resourceId);
      },
      validateOutput: (output) =>
        validateVerificationResult(output, request.pluginId),
    });
  }

  private async executeCapability<TInput, TOutput>(
    options: ExecuteCapabilityOptions<TInput, TOutput>,
  ): Promise<PluginExecutionResult<TOutput>> {
    const executionId = this.idFactory();
    const started = this.now();
    const startedAt = started.toISOString();
    const warnings: string[] = [];
    const { request, capability } = options;
    let pluginVersion = "unknown";

    const finish = async (
      status: PluginExecutionStatus,
      payload: {
        data?: TOutput;
        error?: SerializedPluginExecutionError;
        extraWarnings?: string[];
      },
    ): Promise<PluginExecutionResult<TOutput>> => {
      const finished = this.now();
      const finishedAt = finished.toISOString();
      const durationMs = Math.max(0, finished.getTime() - started.getTime());
      const allWarnings = redactSecrets([
        ...warnings,
        ...(payload.extraWarnings ?? []),
      ]);

      const result = (
        status === "succeeded"
          ? {
              executionId,
              pluginId: request.pluginId,
              pluginVersion,
              capability,
              status,
              startedAt,
              finishedAt,
              durationMs,
              warnings: allWarnings,
              data: redactSecrets(payload.data as TOutput),
            }
          : {
              executionId,
              pluginId: request.pluginId,
              pluginVersion,
              capability,
              status,
              startedAt,
              finishedAt,
              durationMs,
              warnings: allWarnings,
              error: redactSecrets(
                payload.error ??
                  this.makeError(
                    "execution_failed",
                    "Unknown execution failure",
                    request.pluginId,
                    capability,
                  ),
              ),
            }
      ) as PluginExecutionResult<TOutput>;

      await this.emitEvent(result, request);
      return result;
    };

    try {
      options.validateInput();
    } catch (error) {
      return finish("failed", {
        error: this.toSerializedError(
          error,
          "validation_failed",
          request.pluginId,
          capability,
        ),
      });
    }

    const plugin = this.registry.get(request.pluginId);
    if (!plugin) {
      return finish("failed", {
        error: this.makeError(
          "not_found",
          `Plugin not found: ${request.pluginId}`,
          request.pluginId,
          capability,
        ),
      });
    }

    pluginVersion = plugin.manifest.version;

    if (!plugin.manifest.capabilities.includes(capability)) {
      return finish("failed", {
        error: this.makeError(
          "capability_unsupported",
          `Plugin "${request.pluginId}" does not declare capability "${capability}"`,
          request.pluginId,
          capability,
        ),
      });
    }

    const handlerKey = CAPABILITY_HANDLER_KEYS[capability];
    if (typeof plugin[handlerKey] !== "function") {
      return finish("failed", {
        error: this.makeError(
          "missing_handler",
          `Plugin "${request.pluginId}" is missing handler for capability "${capability}"`,
          request.pluginId,
          capability,
        ),
      });
    }

    const requiredPermissions = [
      ...this.capabilityPermissions[capability],
    ] as PluginPermission[];

    for (const permission of requiredPermissions) {
      if (!plugin.manifest.permissions.includes(permission)) {
        return finish("failed", {
          error: this.makeError(
            "permission_denied",
            `Plugin "${request.pluginId}" does not declare required permission "${permission}" for capability "${capability}"`,
            request.pluginId,
            capability,
          ),
        });
      }
    }

    let granted: readonly PluginPermission[];
    try {
      granted = await this.permissionResolver.resolve({
        pluginId: request.pluginId,
        capability,
        actor: request.actor,
        projectId: request.projectId,
        environmentId: request.environmentId,
        resourceId: request.resourceId,
      });
    } catch (error) {
      return finish("failed", {
        error: this.toSerializedError(
          error,
          "permission_denied",
          request.pluginId,
          capability,
        ),
      });
    }

    for (const permission of requiredPermissions) {
      if (!granted.includes(permission)) {
        return finish("failed", {
          error: this.makeError(
            "permission_denied",
            `Permission "${permission}" is required for capability "${capability}" but was not granted`,
            request.pluginId,
            capability,
          ),
        });
      }
    }

    if (options.runApplyGuards) {
      try {
        options.runApplyGuards();
      } catch (error) {
        return finish("failed", {
          error: this.toSerializedError(
            error,
            "approval_invalid",
            request.pluginId,
            capability,
          ),
        });
      }
    }

    const timeoutMs =
      request.timeoutMs ?? DEFAULT_PLUGIN_TIMEOUTS[capability];
    const { signal, cleanup, abortReason } = this.createInvocationSignal(
      request.signal,
      timeoutMs,
    );

    if (signal.aborted) {
      cleanup();
      const mapped = this.mapAbort(
        abortReason,
        signal.reason,
        request.pluginId,
        capability,
      );
      return finish(mapped.status, { error: mapped.error });
    }

    const invokePromise = this.runtime.invoke<TInput, TOutput>({
      pluginId: request.pluginId,
      capability,
      input: options.input,
      signal,
    });
    const abortWait = this.waitForAbort(signal);

    try {
      const output = await Promise.race([invokePromise, abortWait.promise]);

      try {
        options.validateOutput(output);
      } catch (error) {
        return finish("failed", {
          error: this.toSerializedError(
            error,
            "validation_failed",
            request.pluginId,
            capability,
          ),
        });
      }

      return finish("succeeded", { data: output });
    } catch (error) {
      if (signal.aborted || abortReason.kind !== "none") {
        const mapped = this.mapAbort(
          abortReason,
          signal.reason ?? error,
          request.pluginId,
          capability,
        );
        return finish(mapped.status, { error: mapped.error });
      }
      return finish("failed", {
        error: this.normalizeHandlerError(error, request.pluginId, capability),
      });
    } finally {
      abortWait.dispose();
      // In-process handlers are not preemptively stopped; swallow late settle.
      void invokePromise.then(
        () => undefined,
        () => undefined,
      );
      cleanup();
    }
  }

  private assertRequestBase(request: PluginExecutionRequestBase): void {
    if (typeof request.pluginId !== "string" || request.pluginId.trim() === "") {
      throw new PluginValidationError("request.pluginId must be a non-empty string");
    }
    if (
      request.actor === null ||
      typeof request.actor !== "object" ||
      typeof request.actor.id !== "string" ||
      request.actor.id.trim() === ""
    ) {
      throw new PluginValidationError(
        "request.actor.id must be a non-empty string",
        { pluginId: request.pluginId },
      );
    }
    const allowedTypes = new Set(["user", "mcp_agent", "system"]);
    if (!allowedTypes.has(request.actor.type)) {
      throw new PluginValidationError(
        `request.actor.type has unsupported value "${String(request.actor.type)}"`,
        { pluginId: request.pluginId },
      );
    }
    if (
      request.timeoutMs !== undefined &&
      (typeof request.timeoutMs !== "number" ||
        !Number.isFinite(request.timeoutMs) ||
        request.timeoutMs <= 0)
    ) {
      throw new PluginValidationError(
        "request.timeoutMs must be a positive number when provided",
        { pluginId: request.pluginId },
      );
    }
  }

  private assertMatchingPluginId(requestId: string, contextId: string): void {
    if (requestId !== contextId) {
      throw new PluginValidationError(
        `request.pluginId "${requestId}" does not match context.pluginId "${contextId}"`,
        { pluginId: requestId },
      );
    }
  }

  private assertResourceBinding(
    request: PluginExecutionRequestBase,
    bindingResourceId: string,
  ): void {
    if (
      request.resourceId !== undefined &&
      request.resourceId !== bindingResourceId
    ) {
      throw new PluginValidationError(
        `request.resourceId "${request.resourceId}" does not match resource binding "${bindingResourceId}"`,
        { pluginId: request.pluginId },
      );
    }
  }

  private waitForAbort(signal: AbortSignal): {
    promise: Promise<never>;
    dispose: () => void;
  } {
    let listener: (() => void) | undefined;
    const promise = new Promise<never>((_resolve, reject) => {
      if (signal.aborted) {
        reject(signal.reason ?? new Error("aborted"));
        return;
      }
      listener = () => {
        reject(signal.reason ?? new Error("aborted"));
      };
      signal.addEventListener("abort", listener, { once: true });
    });
    return {
      promise,
      dispose: () => {
        if (listener) {
          signal.removeEventListener("abort", listener);
          listener = undefined;
        }
      },
    };
  }

  private createInvocationSignal(
    external: AbortSignal | undefined,
    timeoutMs: number,
  ): {
    signal: AbortSignal;
    cleanup: () => void;
    abortReason: { kind: "external" | "timeout" | "none" };
  } {
    const controller = new AbortController();
    const abortReason: { kind: "external" | "timeout" | "none" } = {
      kind: "none",
    };

    const onExternalAbort = (): void => {
      if (!controller.signal.aborted) {
        abortReason.kind = "external";
        controller.abort(external?.reason ?? new Error("cancelled"));
      }
    };

    let timer: ReturnType<typeof setTimeout> | undefined;
    if (external?.aborted) {
      abortReason.kind = "external";
      controller.abort(external.reason ?? new Error("cancelled"));
    } else {
      external?.addEventListener("abort", onExternalAbort, { once: true });
      timer = setTimeout(() => {
        if (!controller.signal.aborted) {
          abortReason.kind = "timeout";
          controller.abort(new Error("timeout"));
        }
      }, timeoutMs);
    }

    const cleanup = (): void => {
      if (timer !== undefined) {
        clearTimeout(timer);
      }
      external?.removeEventListener("abort", onExternalAbort);
    };

    return { signal: controller.signal, cleanup, abortReason };
  }

  private makeError(
    code: PluginExecutionErrorCode,
    message: string,
    pluginId: string,
    capability: PluginCapability,
    details?: Record<string, unknown>,
  ): SerializedPluginExecutionError {
    return {
      code,
      message,
      pluginId,
      capability,
      retryable: code === "timeout" || code === "execution_failed",
      ...(details !== undefined ? { details: redactSecrets(details) } : {}),
    };
  }

  private mapAbort(
    abortReason: { kind: "external" | "timeout" | "none" },
    reason: unknown,
    pluginId: string,
    capability: PluginCapability,
  ): {
    status: "cancelled" | "timed_out";
    error: SerializedPluginExecutionError;
  } {
    const details =
      reason === undefined
        ? undefined
        : (redactSecrets({ reason }) as Record<string, unknown>);

    if (abortReason.kind === "timeout") {
      return {
        status: "timed_out",
        error: this.makeError(
          "timeout",
          "Plugin capability execution timed out",
          pluginId,
          capability,
          details,
        ),
      };
    }
    return {
      status: "cancelled",
      error: this.makeError(
        "cancelled",
        "Plugin capability execution was cancelled",
        pluginId,
        capability,
        details,
      ),
    };
  }

  private normalizeHandlerError(
    error: unknown,
    pluginId: string,
    capability: PluginCapability,
  ): SerializedPluginExecutionError {
    if (error instanceof PluginError) {
      const code = this.mapPluginErrorCode(error.code);
      return this.makeError(code, error.message, pluginId, capability, {
        name: error.name,
        pluginId: error.pluginId ?? pluginId,
        ...(error instanceof PluginExecutionError
          ? { capability: error.capability }
          : { capability }),
      });
    }

    if (typeof error === "string") {
      return this.makeError("execution_failed", error, pluginId, capability);
    }

    if (error instanceof Error) {
      return this.makeError(
        "execution_failed",
        error.message || "Plugin handler failed",
        pluginId,
        capability,
        {
          name: error.name,
          cause: error.cause,
        },
      );
    }

    return this.makeError(
      "execution_failed",
      "Plugin handler failed with a non-error value",
      pluginId,
      capability,
      redactSecrets(error) as Record<string, unknown>,
    );
  }

  private mapPluginErrorCode(
    code: PluginError["code"],
  ): PluginExecutionErrorCode {
    switch (code) {
      case "not_found":
        return "not_found";
      case "capability_unsupported":
        return "capability_unsupported";
      case "validation_failed":
        return "validation_failed";
      case "version_mismatch":
        return "validation_failed";
      case "execution_failed":
      default:
        return "execution_failed";
    }
  }

  private toSerializedError(
    error: unknown,
    fallbackCode: PluginExecutionErrorCode,
    pluginId: string,
    capability: PluginCapability,
  ): SerializedPluginExecutionError {
    if (error instanceof PluginError) {
      const mapped =
        fallbackCode === "approval_invalid"
          ? "approval_invalid"
          : this.mapPluginErrorCode(error.code) === "validation_failed"
            ? fallbackCode
            : this.mapPluginErrorCode(error.code);
      return this.makeError(mapped, error.message, pluginId, capability);
    }
    if (error instanceof Error) {
      return this.makeError(fallbackCode, error.message, pluginId, capability);
    }
    return this.makeError(fallbackCode, String(error), pluginId, capability);
  }

  private async emitEvent(
    result: PluginExecutionResult<unknown>,
    request: PluginExecutionRequestBase,
  ): Promise<void> {
    if (!this.eventSink) {
      return;
    }

    try {
      await this.eventSink.record({
        executionId: result.executionId,
        pluginId: result.pluginId,
        pluginVersion: result.pluginVersion,
        capability: result.capability,
        status: result.status,
        startedAt: result.startedAt,
        finishedAt: result.finishedAt,
        durationMs: result.durationMs,
        actor: request.actor,
        projectId: request.projectId,
        environmentId: request.environmentId,
        resourceId: request.resourceId,
        reason: request.reason,
        errorCode: result.error?.code,
        errorMessage: result.error?.message,
        warningCount: result.warnings.length,
      });
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Plugin execution event sink failed";
      result.warnings.push(
        redactSecrets(`event_sink_failed: ${message}`),
      );
    }
  }
}
