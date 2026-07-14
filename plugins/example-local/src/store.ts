import {
  LOCAL_SERVICE_RESOURCE_TYPE,
  LOCAL_SERVICE_SCHEMA_VERSION,
} from "./manifest.js";

export interface LocalServiceRecord {
  providerResourceId: string;
  name: string;
  port: number;
  status: "ready" | "degraded";
}

const FIXTURES: readonly LocalServiceRecord[] = [
  {
    providerResourceId: "local-api",
    name: "Local API",
    port: 3000,
    status: "ready",
  },
  {
    providerResourceId: "local-worker",
    name: "Local Worker",
    port: 3001,
    status: "ready",
  },
];

/** Deterministic in-memory store. Isolated from the host filesystem. */
export class ExampleLocalStore {
  private readonly services = new Map<string, LocalServiceRecord>();

  constructor() {
    this.reset();
  }

  reset(): void {
    this.services.clear();
    for (const fixture of FIXTURES) {
      this.services.set(fixture.providerResourceId, { ...fixture });
    }
  }

  list(): LocalServiceRecord[] {
    return [...this.services.values()].map((service) => ({ ...service }));
  }

  get(providerResourceId: string): LocalServiceRecord | undefined {
    const service = this.services.get(providerResourceId);
    return service ? { ...service } : undefined;
  }

  setPort(providerResourceId: string, port: number): LocalServiceRecord {
    const current = this.services.get(providerResourceId);
    if (!current) {
      throw new Error(`Unknown local service: ${providerResourceId}`);
    }
    const next = { ...current, port };
    this.services.set(providerResourceId, next);
    return { ...next };
  }

  toDiscovered(service: LocalServiceRecord) {
    return {
      providerResourceId: service.providerResourceId,
      resourceType: LOCAL_SERVICE_RESOURCE_TYPE,
      name: service.name,
      metadata: {
        port: service.port,
        status: service.status,
      },
      schemaVersion: LOCAL_SERVICE_SCHEMA_VERSION,
    };
  }
}
