#!/usr/bin/env node

import {
  DaemonClient,
  daemonCredentialStorePath,
  LocalClientCredentialStore,
  daemonEndpointPath,
  defaultRayvanDataDir,
  defaultRayvanRuntimeDir,
} from "@rayvan/daemon-client";
import { BUILT_IN_LOCAL_CLIENT_IDS, DaemonMethods } from "@rayvan/daemon-contracts";

import { acquireDaemonLock, releaseDaemonLock } from "./lock.js";
import { DaemonRuntime } from "./runtime.js";
import { DaemonIpcServer } from "./server.js";

type Command = "serve" | "status" | "stop" | "diagnostics";

interface CliOptions {
  command: Command;
  dataDir: string;
  runtimeDir: string;
  endpoint: string;
}

async function run(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));

  if (options.command === "serve") {
    await serve(options);
    return;
  }

  const client = new DaemonClient({
    endpoint: options.endpoint,
    clientType: "cli",
    clientVersion: "0.0.1",
    clientId: BUILT_IN_LOCAL_CLIENT_IDS.cli,
    clientCredential:
      new LocalClientCredentialStore(
        daemonCredentialStorePath(options.dataDir),
      ).resolve(BUILT_IN_LOCAL_CLIENT_IDS.cli) ?? undefined,
  });
  try {
    await client.connect();
    const result =
      options.command === "stop"
        ? await client.call(DaemonMethods.shutdown)
        : await client.call(
            options.command === "status"
              ? DaemonMethods.status
              : DaemonMethods.diagnostics,
          );
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  } finally {
    await client.close();
  }
}

async function serve(options: CliOptions): Promise<void> {
  const lock = await acquireDaemonLock({
    runtimeDir: options.runtimeDir,
    endpoint: options.endpoint,
    dataDir: options.dataDir,
  });

  if (lock.status === "reused") {
    process.stdout.write(`${JSON.stringify(lock.info, null, 2)}\n`);
    return;
  }
  if (lock.status === "incompatible") {
    throw new Error(lock.reason);
  }

  let runtime: DaemonRuntime;
  try {
    runtime = new DaemonRuntime({
      dataDir: options.dataDir,
      runtimeDir: options.runtimeDir,
      endpoint: options.endpoint,
    });
  } catch (error) {
    releaseDaemonLock(options.runtimeDir);
    throw error;
  }
  const server = new DaemonIpcServer({ runtime });
  let stopped = false;

  const stop = async () => {
    if (stopped) {
      return;
    }
    stopped = true;
    try {
      await server.close();
    } finally {
      try {
        runtime.close();
      } finally {
        releaseDaemonLock(options.runtimeDir);
      }
    }
  };

  const signalPromise = new Promise<void>((resolve) => {
    const onSignal = () => {
      void stop().then(resolve);
    };
    process.once("SIGINT", onSignal);
    process.once("SIGTERM", onSignal);
  });

  try {
    await server.start();
    await signalPromise;
  } catch (error) {
    await stop();
    throw error;
  }
}

function parseArgs(args: string[]): CliOptions {
  const command = (args[0] ?? "serve") as Command;
  if (!["serve", "status", "stop", "diagnostics"].includes(command)) {
    throw new Error(
      `Unknown command "${command}". Expected serve, status, stop, or diagnostics.`,
    );
  }

  const dataDir = option(args, "--data-dir") ?? defaultRayvanDataDir();
  const runtimeDir = option(args, "--runtime-dir") ?? defaultRayvanRuntimeDir();
  const endpoint = option(args, "--endpoint") ?? daemonEndpointPath(runtimeDir);
  return { command, dataDir, runtimeDir, endpoint };
}

function option(args: string[], name: string): string | undefined {
  const index = args.indexOf(name);
  if (index === -1) {
    return undefined;
  }
  const value = args[index + 1];
  if (!value || value.startsWith("--")) {
    throw new Error(`Missing value for ${name}`);
  }
  return value;
}

void run().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`rayvand: ${message}\n`);
  process.exitCode = 1;
});
