export type McpTransport = "stdio";

export interface TransportConfig {
  kind: McpTransport;
}

export const DEFAULT_TRANSPORT: TransportConfig = {
  kind: "stdio",
};
