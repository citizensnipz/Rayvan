export class PluginClientError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "PluginClientError";
    this.code = code;
  }
}

export class PluginTimeoutError extends PluginClientError {
  constructor(message = "Plugin request timed out") {
    super("timeout", message);
    this.name = "PluginTimeoutError";
  }
}
