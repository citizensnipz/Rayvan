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
  createDaemonPluginIntegrationsGateway,
  createDevPluginIntegrationsGateway,
} from "../../lib/plugins/index.js";
import type { PluginIntegrationsGateway } from "../../lib/plugins/index.js";

interface IntegrationsContextValue {
  gateway: PluginIntegrationsGateway;
}

const IntegrationsContext = createContext<IntegrationsContextValue | null>(null);

interface IntegrationsProviderProps extends PropsWithChildren {
  /**
   * Inject a gateway (e.g. a fresh dev fixture instance, or a test double)
   * so callers never share singleton state. Defaults to daemon-first when
   * connected, otherwise a fresh development fixture gateway.
   */
  gateway?: PluginIntegrationsGateway;
}

export function IntegrationsProvider({
  gateway,
  children,
}: IntegrationsProviderProps) {
  const daemon = useOptionalDaemonConnection();
  const instance = useMemo<PluginIntegrationsGateway>(() => {
    if (gateway) {
      return gateway;
    }
    return preferDaemonGateways(daemon?.connected === true)
      ? createDaemonPluginIntegrationsGateway()
      : createDevPluginIntegrationsGateway();
  }, [gateway, daemon?.connected]);

  return (
    <IntegrationsContext.Provider value={{ gateway: instance }}>
      {children}
    </IntegrationsContext.Provider>
  );
}

export function useIntegrationsGateway(): PluginIntegrationsGateway {
  const context = useContext(IntegrationsContext);
  if (!context) {
    throw new Error(
      "useIntegrationsGateway must be used within IntegrationsProvider",
    );
  }
  return context.gateway;
}
