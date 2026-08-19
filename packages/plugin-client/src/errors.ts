export class PluginTransportError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PluginTransportError";
  }
}

export class PluginHostError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PluginHostError";
  }
}
