import { chmodSync, existsSync, unlinkSync } from "node:fs";
import { createServer, type Server, type Socket } from "node:net";

import {
  BUILT_IN_PERMISSION_PROFILES,
  DAEMON_RPC_ERROR_CODES,
  DaemonMethods,
  encodeFrame,
  FrameDecoder,
  type DaemonErrorCode,
  type DaemonEventNotification,
  type DaemonHandshakeRequest,
  type DaemonRequestEnvelope,
  type DaemonResponseEnvelope,
  type LocalClientRecord,
  type RayvanActor,
} from "@rayvan/daemon-contracts";

import type { SessionContext } from "./auth/session.js";
import { permissionsForProfile } from "./auth/session.js";
import { DaemonAppError, toDaemonError } from "./errors.js";
import { DaemonRuntime } from "./runtime.js";

export interface DaemonIpcServerOptions {
  runtime: DaemonRuntime;
}

export class DaemonIpcServer {
  private readonly runtime: DaemonRuntime;
  private readonly sockets = new Set<Socket>();
  private server: Server | null = null;
  private closing: Promise<void> | null = null;

  constructor(options: DaemonIpcServerOptions) {
    this.runtime = options.runtime;
  }

  async start(): Promise<void> {
    if (this.server) {
      throw new Error("Daemon IPC server is already running");
    }

    this.runtime.recoverIncompleteOperations();
    removeUnixSocket(this.runtime.endpoint);

    const server = createServer((socket) => this.accept(socket));
    this.server = server;
    await new Promise<void>((resolve, reject) => {
      const onError = (error: Error) => {
        server.off("listening", onListening);
        this.server = null;
        reject(error);
      };
      const onListening = () => {
        server.off("error", onError);
        resolve();
      };
      server.once("error", onError);
      server.once("listening", onListening);
      server.listen(this.runtime.endpoint);
    });

    if (isUnixSocket(this.runtime.endpoint)) {
      chmodSync(this.runtime.endpoint, 0o600);
    }
  }

  close(): Promise<void> {
    if (this.closing) {
      return this.closing;
    }

    this.closing = this.closeInner();
    return this.closing;
  }

  private async closeInner(): Promise<void> {
    const server = this.server;
    this.server = null;

    for (const socket of this.sockets) {
      socket.end();
    }

    if (server) {
      await new Promise<void>((resolve) => {
        server.close(() => resolve());
      });
    }

    for (const socket of this.sockets) {
      socket.destroy();
    }
    this.sockets.clear();
    removeUnixSocket(this.runtime.endpoint);
  }

  private accept(socket: Socket): void {
    this.sockets.add(socket);
    const decoder = new FrameDecoder();
    let session: SessionContext | undefined;
    let processing = Promise.resolve();

    const unsubscribe = this.runtime.events.subscribe((event) => {
      if (
        !session?.subscribed ||
        socket.destroyed ||
        !canReceiveEvent(session, event)
      ) {
        return;
      }
      const notification: DaemonEventNotification = {
        jsonrpc: "2.0",
        method: "daemon.event",
        params: event,
      };
      socket.write(encodeFrame(notification));
    });

    const cleanup = () => {
      unsubscribe();
      this.sockets.delete(socket);
      if (session) {
        this.runtime.unregisterSession(session.sessionId);
        session = undefined;
      }
    };

    socket.once("close", cleanup);
    socket.on("error", () => {
      // Connection errors are isolated to this client.
    });
    socket.on("data", (chunk) => {
      let messages: unknown[];
      try {
        messages = decoder.push(chunk);
      } catch (error) {
        this.writeError(socket, null, error);
        socket.end();
        return;
      }

      for (const message of messages) {
        processing = processing
          .then(async () => {
            session = await this.handleMessage(socket, message, session);
          })
          .catch((error) => {
            this.writeError(socket, requestId(message), error);
          });
      }
    });
  }

  private async handleMessage(
    socket: Socket,
    message: unknown,
    session: SessionContext | undefined,
  ): Promise<SessionContext | undefined> {
    const request = parseRequest(message);

    if (!session) {
      if (request.method !== DaemonMethods.handshake) {
        this.writeError(
          socket,
          request.id,
          new DaemonAppError(
            "UNAUTHENTICATED",
            "system.handshake must be the first request",
          ),
        );
        return undefined;
      }

      const params = asHandshake(request.params);
      const response = await this.runtime.handshake(params);
      const authenticatedSession = deriveSession(this.runtime, params, response);
      this.runtime.registerSession(authenticatedSession);
      this.writeResult(socket, request.id, response);
      return authenticatedSession;
    }

    if (request.method === DaemonMethods.handshake) {
      throw new DaemonAppError(
        "VALIDATION_FAILED",
        "Handshake has already completed for this connection",
      );
    }

    const result = await this.runtime.dispatch(session, request.method, request.params);
    this.writeResult(socket, request.id, result);
    return session;
  }

  private writeResult(socket: Socket, id: string, result: unknown): void {
    if (socket.destroyed) {
      return;
    }
    const response: DaemonResponseEnvelope = {
      jsonrpc: "2.0",
      id,
      result,
    };
    socket.write(encodeFrame(response));
  }

  private writeError(socket: Socket, id: string | null, error: unknown): void {
    if (socket.destroyed) {
      return;
    }
    const serialized = toDaemonError(error);
    const response: DaemonResponseEnvelope = {
      jsonrpc: "2.0",
      id,
      error: {
        code: rpcErrorCode(serialized.code),
        message: serialized.message,
        data: serialized,
      },
    };
    socket.write(encodeFrame(response));
  }
}

function deriveSession(
  runtime: DaemonRuntime,
  request: DaemonHandshakeRequest,
  response: Awaited<ReturnType<DaemonRuntime["handshake"]>>,
): SessionContext {
  const client = response.authenticatedClientId
    ? (runtime.control.getClient(response.authenticatedClientId) ?? undefined)
    : undefined;

  if (response.authenticatedClientId && !client) {
    throw new DaemonAppError(
      "CLIENT_NOT_REGISTERED",
      "Authenticated client no longer exists",
    );
  }

  return {
    sessionId: response.sessionId,
    clientType: request.clientType,
    clientVersion: request.clientVersion,
    client,
    actor: actorFor(request.clientType, client),
    permissions: client
      ? permissionsForProfile(client.permissionProfileId)
      : new Set(BUILT_IN_PERMISSION_PROFILES.administrator),
    projectScopes: client?.projectScopes ?? "*",
    environmentScopes: client?.environmentScopes ?? "*",
    subscribed: false,
  };
}

function actorFor(
  clientType: DaemonHandshakeRequest["clientType"],
  client: LocalClientRecord | undefined,
): RayvanActor {
  if (client) {
    if (client.type === "mcp") {
      return { type: "mcp_client", id: client.id, displayName: client.name };
    }
    if (client.type === "desktop") {
      return { type: "desktop", id: client.id, displayName: client.name };
    }
    return { type: "user", id: client.id, displayName: client.name };
  }

  return {
    type: clientType === "desktop" ? "desktop" : "system",
    id: `${clientType}_bootstrap`,
    displayName: `${clientType} bootstrap`,
  };
}

function canReceiveEvent(
  session: SessionContext,
  event: DaemonEventNotification["params"],
): boolean {
  if (session.client?.permissionProfileId === "administrator") {
    return true;
  }
  if (
    event.projectId &&
    session.projectScopes !== "*" &&
    !session.projectScopes.includes(event.projectId)
  ) {
    return false;
  }
  const environmentId = event.payload.environmentId;
  if (
    typeof environmentId === "string" &&
    session.environmentScopes !== undefined &&
    session.environmentScopes !== "*" &&
    !session.environmentScopes.includes(environmentId)
  ) {
    return false;
  }
  return true;
}

function parseRequest(message: unknown): DaemonRequestEnvelope {
  if (!message || typeof message !== "object" || Array.isArray(message)) {
    throw new DaemonAppError("VALIDATION_FAILED", "Invalid JSON-RPC request");
  }
  const request = message as Partial<DaemonRequestEnvelope>;
  if (
    request.jsonrpc !== "2.0" ||
    typeof request.id !== "string" ||
    request.id.length === 0 ||
    typeof request.method !== "string" ||
    request.method.length === 0
  ) {
    throw new DaemonAppError("VALIDATION_FAILED", "Invalid JSON-RPC request");
  }
  return request as DaemonRequestEnvelope;
}

function asHandshake(params: unknown): DaemonHandshakeRequest {
  if (!params || typeof params !== "object" || Array.isArray(params)) {
    throw new DaemonAppError("VALIDATION_FAILED", "Invalid handshake params");
  }
  const value = params as Partial<DaemonHandshakeRequest>;
  if (
    typeof value.protocolVersion !== "string" ||
    typeof value.clientType !== "string" ||
    typeof value.clientVersion !== "string"
  ) {
    throw new DaemonAppError("VALIDATION_FAILED", "Invalid handshake params");
  }
  return value as DaemonHandshakeRequest;
}

function requestId(message: unknown): string | null {
  if (!message || typeof message !== "object" || Array.isArray(message)) {
    return null;
  }
  const id = (message as { id?: unknown }).id;
  return typeof id === "string" ? id : null;
}

function rpcErrorCode(code: DaemonErrorCode): number {
  switch (code) {
    case "UNAUTHENTICATED":
    case "CLIENT_NOT_REGISTERED":
    case "CLIENT_REVOKED":
      return DAEMON_RPC_ERROR_CODES.UNAUTHENTICATED;
    case "PERMISSION_DENIED":
    case "PROJECT_SCOPE_DENIED":
    case "ENVIRONMENT_SCOPE_DENIED":
      return DAEMON_RPC_ERROR_CODES.PERMISSION_DENIED;
    case "DAEMON_VERSION_MISMATCH":
      return DAEMON_RPC_ERROR_CODES.VERSION_MISMATCH;
    default:
      return DAEMON_RPC_ERROR_CODES.APPLICATION;
  }
}

function isUnixSocket(endpoint: string): boolean {
  return !endpoint.startsWith("\\\\.\\pipe\\");
}

function removeUnixSocket(endpoint: string): void {
  if (isUnixSocket(endpoint) && existsSync(endpoint)) {
    unlinkSync(endpoint);
  }
}
