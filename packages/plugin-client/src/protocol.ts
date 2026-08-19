import type { PluginCapability } from "@rayvan/plugin-sdk";

/** Capability-aligned JSON-RPC 2.0 methods for OOP plugins. */
export const PLUGIN_RPC_METHODS = [
  "initialize",
  "shutdown",
  "authenticate",
  "discover",
  "inspect",
  "plan",
  "apply",
  "verify",
  "evaluate_findings",
  "cancel",
] as const;

export type PluginRpcMethod = (typeof PLUGIN_RPC_METHODS)[number];

export const PLUGIN_PROTOCOL_VERSION = "1";

export interface JsonRpcRequest<TParams = unknown> {
  jsonrpc: "2.0";
  id: string | number;
  method: PluginRpcMethod | string;
  params?: TParams;
}

export interface JsonRpcSuccess<TResult = unknown> {
  jsonrpc: "2.0";
  id: string | number | null;
  result: TResult;
}

export interface JsonRpcErrorObject {
  code: number;
  message: string;
  data?: unknown;
}

export interface JsonRpcFailure {
  jsonrpc: "2.0";
  id: string | number | null;
  error: JsonRpcErrorObject;
}

export type JsonRpcResponse<TResult = unknown> =
  | JsonRpcSuccess<TResult>
  | JsonRpcFailure;

export function isJsonRpcResponse(value: unknown): value is JsonRpcResponse {
  if (value === null || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  return record.jsonrpc === "2.0" && ("result" in record || "error" in record);
}

export function capabilityToRpcMethod(
  capability: PluginCapability,
): PluginRpcMethod {
  return capability;
}

export interface PluginInitializeParams {
  protocolVersion: string;
  pluginId: string;
  hostCapabilities?: string[];
}

export interface PluginInitializeResult {
  protocolVersion: string;
  pluginId: string;
  capabilities: PluginCapability[];
}
