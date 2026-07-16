import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type PropsWithChildren,
} from "react";
import type { DaemonEvent, DaemonStatus } from "@rayvan/daemon-contracts";

import {
  getDaemonStatus,
  listenDaemonEvents,
  reconnectDaemon,
  type DaemonStatusSnapshot,
} from "./client.js";

export interface DaemonConnectionState {
  loading: boolean;
  connected: boolean;
  snapshot: DaemonStatusSnapshot | null;
  status: DaemonStatus | null;
  lastError: string | null;
  lastEvent: DaemonEvent | null;
  refresh: () => Promise<void>;
  reconnect: () => Promise<void>;
}

const DaemonConnectionContext = createContext<DaemonConnectionState | null>(
  null,
);

export function DaemonConnectionProvider({ children }: PropsWithChildren) {
  const [loading, setLoading] = useState(true);
  const [snapshot, setSnapshot] = useState<DaemonStatusSnapshot | null>(null);
  const [lastEvent, setLastEvent] = useState<DaemonEvent | null>(null);

  const refresh = useCallback(async () => {
    try {
      const next = await getDaemonStatus();
      setSnapshot(next);
    } catch (error) {
      setSnapshot({
        connected: false,
        endpoint: "",
        spawned: false,
        lastError:
          error instanceof Error ? error.message : "Failed to read daemon status",
      });
    } finally {
      setLoading(false);
    }
  }, []);

  const reconnect = useCallback(async () => {
    setLoading(true);
    try {
      const next = await reconnectDaemon();
      setSnapshot(next);
    } catch (error) {
      setSnapshot((previous) => ({
        connected: false,
        endpoint: previous?.endpoint ?? "",
        spawned: previous?.spawned ?? false,
        lastError:
          error instanceof Error
            ? error.message
            : "Failed to reconnect to daemon",
      }));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    let unlisten: (() => void) | undefined;
    void listenDaemonEvents((event) => {
      setLastEvent(event);
      if (event.type === "daemon_status_changed") {
        void refresh();
      }
    }).then((fn) => {
      unlisten = fn;
    });
    const timer = window.setInterval(() => {
      void refresh();
    }, 15_000);
    return () => {
      window.clearInterval(timer);
      unlisten?.();
    };
  }, [refresh]);

  const value = useMemo<DaemonConnectionState>(
    () => ({
      loading,
      connected: snapshot?.connected === true,
      snapshot,
      status: snapshot?.status ?? null,
      lastError: snapshot?.lastError ?? null,
      lastEvent,
      refresh,
      reconnect,
    }),
    [loading, snapshot, lastEvent, refresh, reconnect],
  );

  return (
    <DaemonConnectionContext.Provider value={value}>
      {children}
    </DaemonConnectionContext.Provider>
  );
}

export function useDaemonConnection(): DaemonConnectionState {
  const context = useContext(DaemonConnectionContext);
  if (!context) {
    throw new Error(
      "useDaemonConnection must be used within DaemonConnectionProvider",
    );
  }
  return context;
}

/** Returns null when no provider is mounted (e.g. focused gateway unit tests). */
export function useOptionalDaemonConnection(): DaemonConnectionState | null {
  return useContext(DaemonConnectionContext);
}
