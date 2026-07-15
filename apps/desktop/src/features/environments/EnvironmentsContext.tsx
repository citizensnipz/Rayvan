import {
  createContext,
  useContext,
  useState,
  type PropsWithChildren,
} from "react";

import {
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
   * so callers never share singleton state. Defaults to a fresh
   * development fixture gateway created once per provider mount.
   */
  gateway?: EnvironmentsGateway;
}

export function EnvironmentsProvider({ gateway, children }: EnvironmentsProviderProps) {
  const [instance] = useState<EnvironmentsGateway>(
    () => gateway ?? createDevEnvironmentsGateway(),
  );

  return (
    <EnvironmentsContext.Provider value={{ gateway: instance }}>
      {children}
    </EnvironmentsContext.Provider>
  );
}

export function useEnvironmentsGateway(): EnvironmentsGateway {
  const context = useContext(EnvironmentsContext);
  if (!context) {
    throw new Error("useEnvironmentsGateway must be used within EnvironmentsProvider");
  }
  return context.gateway;
}
