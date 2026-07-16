import {
  createContext,
  useContext,
  useState,
  type PropsWithChildren,
} from "react";

import {
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
   * so callers never share singleton state. Defaults to a fresh
   * development fixture gateway created once per provider mount.
   */
  gateway?: FindingsGateway;
}

export function FindingsProvider({ gateway, children }: FindingsProviderProps) {
  const [instance] = useState<FindingsGateway>(
    () => gateway ?? createDevFindingsGateway(),
  );

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
