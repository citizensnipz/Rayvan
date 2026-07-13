import type { Logger } from "@rayvan/shared";

export interface PluginInitializeContext {
  pluginId: string;
  dataDirectory: string;
  logger: Logger;
}

export interface PluginConnectionContext extends PluginInitializeContext {
  integrationId: string;
}

export interface PluginDiscoveryContext extends PluginConnectionContext {
  projectRootPath?: string;
}

export interface PluginConfigurationContext extends PluginConnectionContext {
  environmentId: string;
}

export interface PluginHealthContext extends PluginConnectionContext {
  environmentId?: string;
}

export interface PluginActionContext extends PluginConnectionContext {
  projectId: string;
  environmentId?: string;
}

export interface PluginExecutionContext extends PluginActionContext {
  approvalId: string;
}
