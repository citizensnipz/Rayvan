import { createConnection, type Socket } from "node:net";

import {
  type DaemonClientType,
  type DaemonEvent,
  type DaemonEventNotification,
  type DaemonHandshakeRequest,
  type DaemonHandshakeResponse,
  type DaemonRequestEnvelope,
  type DaemonResponseEnvelope,
  DAEMON_PROTOCOL_VERSION,
  encodeFrame,
  FrameDecoder,
} from "@rayvan/daemon-contracts";

import { daemonEndpointPath } from "./paths.js";

export interface DaemonClientTransportOptions {
  endpoint?: string;
  clientType: DaemonClientType;
  clientVersion: string;
  clientId?: string;
  clientCredential?: string;
  connectTimeoutMs?: number;
  requestTimeoutMs?: number;
}

type Pending = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
};

export class DaemonIpcTransport {
  private socket: Socket | null = null;
  private readonly decoder = new FrameDecoder();
  private readonly pending = new Map<string, Pending>();
  private readonly eventListeners = new Set<(event: DaemonEvent) => void>();
  private session: DaemonHandshakeResponse | null = null;
  private requestCounter = 0;

  constructor(private readonly options: DaemonClientTransportOptions) {}

  get handshake(): DaemonHandshakeResponse | null {
    return this.session;
  }

  async connect(): Promise<DaemonHandshakeResponse> {
    if (this.socket) {
      throw new Error("Transport already connected");
    }

    const endpoint = this.options.endpoint ?? daemonEndpointPath();
    const socket = await this.openSocket(endpoint);
    this.socket = socket;

    socket.on("data", (chunk) => {
      try {
        const messages = this.decoder.push(chunk);
        for (const message of messages) {
          this.handleMessage(message);
        }
      } catch (error) {
        this.failAll(error instanceof Error ? error : new Error(String(error)));
      }
    });

    socket.on("error", (error) => {
      this.failAll(error);
    });

    socket.on("close", () => {
      this.failAll(new Error("Daemon connection closed"));
      this.socket = null;
      this.session = null;
    });

    const handshake: DaemonHandshakeRequest = {
      protocolVersion: DAEMON_PROTOCOL_VERSION,
      clientType: this.options.clientType,
      clientVersion: this.options.clientVersion,
      clientId: this.options.clientId,
      clientCredential: this.options.clientCredential,
    };

    this.session = (await this.request(
      "system.handshake",
      handshake,
    )) as DaemonHandshakeResponse;
    return this.session;
  }

  async request(method: string, params?: unknown): Promise<unknown> {
    if (!this.socket) {
      throw new Error("Not connected to daemon");
    }
    const id = `req_${++this.requestCounter}_${Date.now()}`;
    const envelope: DaemonRequestEnvelope = {
      jsonrpc: "2.0",
      id,
      method,
      params,
    };

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Timed out waiting for daemon response to ${method}`));
        this.socket?.destroy();
      }, this.options.requestTimeoutMs ?? 30_000);
      this.pending.set(id, { resolve, reject, timer });
      this.socket!.write(encodeFrame(envelope), (error) => {
        if (error) {
          const pending = this.pending.get(id);
          if (pending) clearTimeout(pending.timer);
          this.pending.delete(id);
          reject(error);
        }
      });
    });
  }

  onEvent(listener: (event: DaemonEvent) => void): () => void {
    this.eventListeners.add(listener);
    return () => {
      this.eventListeners.delete(listener);
    };
  }

  async subscribe(eventTypes?: string[]): Promise<void> {
    await this.request("system.subscribe", { eventTypes: eventTypes ?? ["*"] });
  }

  async close(): Promise<void> {
    if (!this.socket) {
      return;
    }
    const socket = this.socket;
    this.socket = null;
    await new Promise<void>((resolve) => {
      socket.end(() => resolve());
    });
  }

  private openSocket(endpoint: string): Promise<Socket> {
    const timeoutMs = this.options.connectTimeoutMs ?? 5_000;
    return new Promise((resolve, reject) => {
      const socket = createConnection(endpoint);
      const timer = setTimeout(() => {
        socket.destroy();
        reject(new Error(`Timed out connecting to daemon at ${endpoint}`));
      }, timeoutMs);

      socket.once("connect", () => {
        clearTimeout(timer);
        resolve(socket);
      });
      socket.once("error", (error) => {
        clearTimeout(timer);
        reject(error);
      });
    });
  }

  private handleMessage(message: unknown): void {
    if (!message || typeof message !== "object") {
      return;
    }
    const record = message as Record<string, unknown>;

    if (record.method === "daemon.event") {
      const notification = message as DaemonEventNotification;
      for (const listener of this.eventListeners) {
        listener(notification.params);
      }
      return;
    }

    const response = message as DaemonResponseEnvelope;
    if (!("id" in response) || response.id === null) {
      return;
    }
    const pending = this.pending.get(String(response.id));
    if (!pending) {
      return;
    }
    this.pending.delete(String(response.id));
    clearTimeout(pending.timer);
    if ("error" in response) {
      const err = new Error(response.error.message);
      (err as Error & { code?: string; data?: unknown }).code =
        response.error.data?.code ?? String(response.error.code);
      (err as Error & { data?: unknown }).data = response.error.data;
      pending.reject(err);
      return;
    }
    pending.resolve(response.result);
  }

  private failAll(error: Error): void {
    for (const [, pending] of this.pending) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.pending.clear();
  }
}
