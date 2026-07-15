import {
  createContext,
  useContext,
  useState,
  type PropsWithChildren,
} from "react";

import { createDevPluginIntegrationsGateway } from "../../lib/plugins/index.js";
import type { PluginIntegrationsGateway } from "../../lib/plugins/index.js";

interface IntegrationsContextValue {
  gateway: PluginIntegrationsGateway;
}

const IntegrationsContext = createContext<IntegrationsContextValue | null>(null);

interface IntegrationsProviderProps extends PropsWithChildren {
  /**
   * Inject a gateway (e.g. a fresh dev fixture instance, or a test double)
   * so callers never share singleton state. Defaults to a fresh
   * development fixture gateway created once per provider mount.
   */
  gateway?: PluginIntegrationsGateway;
}

export function IntegrationsProvider({ gateway, children }: IntegrationsProviderProps) {
  const [instance] = useState<PluginIntegrationsGateway>(
    () => gateway ?? createDevPluginIntegrationsGateway(),
  );

  return (
    <IntegrationsContext.Provider value={{ gateway: instance }}>
      {children}
    </IntegrationsContext.Provider>
  );
}

export function useIntegrationsGateway(): PluginIntegrationsGateway {
  const context = useContext(IntegrationsContext);
  if (!context) {
    throw new Error("useIntegrationsGateway must be used within IntegrationsProvider");
  }
  return context.gateway;
}
