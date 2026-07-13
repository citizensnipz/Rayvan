export interface PluginProcessHandle {
  pluginId: string;
  pid?: number;
}

export interface PluginProcessClient {
  send<TResponse>(request: unknown): Promise<TResponse>;
  ping(): Promise<boolean>;
}
