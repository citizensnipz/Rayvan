export type PluginRequestMethod =
  | "initialize"
  | "testConnection"
  | "discoverResources"
  | "collectConfiguration"
  | "collectHealth"
  | "planAction"
  | "executeAction"
  | "dispose";

export interface PluginRequest<TPayload = unknown> {
  id: string;
  method: PluginRequestMethod;
  payload: TPayload;
}

export interface PluginResponse<TPayload = unknown> {
  id: string;
  ok: boolean;
  payload?: TPayload;
  error?: {
    code: string;
    message: string;
  };
}
