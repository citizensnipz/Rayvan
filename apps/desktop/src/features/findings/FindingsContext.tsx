import {
  createContext,
  useContext,
  useMemo,
  type PropsWithChildren,
} from "react";

import {
  preferDaemonGateways,
  useOptionalDaemonConnection,
} from "../../lib/daemon/index.js";
import {
  createDaemonFindingsGateway,
  createDevFindingsGateway,
  type FindingsGateway,
} from "../../lib/findings/index.js";

interface FindingsContextValue {
  gateway: FindingsGateway;
}

const FindingsContext = createContext<FindingsContextValue | null>(null);

interface FindingsProviderProps extends PropsWithChildren {
  /**
   * Inject a gateway (e.g. a fresh dev fixture instance, or a test double)
   * so callers never share singleton state. Defaults to daemon-first when
   * connected, otherwise a fresh development fixture gateway.
   */
  gateway?: FindingsGateway;
}

export function FindingsProvider({ gateway, children }: FindingsProviderProps) {
  const daemon = useOptionalDaemonConnection();
  const instance = useMemo<FindingsGateway>(() => {
    if (gateway) {
      return gateway;
    }
    return preferDaemonGateways(daemon?.connected === true)
      ? createDaemonFindingsGateway()
      : createDevFindingsGateway();
  }, [gateway, daemon?.connected]);

  return (
    <FindingsContext.Provider value={{ gateway: instance }}>
      {children}
    </FindingsContext.Provider>
  );
}

export function useFindingsGateway(): FindingsGateway {
  const context = useContext(FindingsContext);
  if (!context) {
    throw new Error("useFindingsGateway must be used within FindingsProvider");
  }
  return context.gateway;
}
