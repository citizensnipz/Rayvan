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
  createDaemonEnvironmentsGateway,
  createDevEnvironmentsGateway,
  type EnvironmentsGateway,
} from "../../lib/environments/index.js";

interface EnvironmentsContextValue {
  gateway: EnvironmentsGateway;
}

const EnvironmentsContext = createContext<EnvironmentsContextValue | null>(null);

interface EnvironmentsProviderProps extends PropsWithChildren {
  /**
   * Inject a gateway (e.g. a fresh dev fixture instance, or a test double)
   * so callers never share singleton state. Defaults to daemon-first when
   * connected, otherwise a fresh development fixture gateway.
   */
  gateway?: EnvironmentsGateway;
}

export function EnvironmentsProvider({
  gateway,
  children,
}: EnvironmentsProviderProps) {
  const daemon = useOptionalDaemonConnection();
  const instance = useMemo<EnvironmentsGateway>(() => {
    if (gateway) {
      return gateway;
    }
    return preferDaemonGateways(daemon?.connected === true)
      ? createDaemonEnvironmentsGateway()
      : createDevEnvironmentsGateway();
  }, [gateway, daemon?.connected]);

  return (
    <EnvironmentsContext.Provider value={{ gateway: instance }}>
      {children}
    </EnvironmentsContext.Provider>
  );
}

export function useEnvironmentsGateway(): EnvironmentsGateway {
  const context = useContext(EnvironmentsContext);
  if (!context) {
    throw new Error(
      "useEnvironmentsGateway must be used within EnvironmentsProvider",
    );
  }
  return context.gateway;
}
