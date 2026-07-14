import { PluginConnectionNotFoundError, PluginDomainError } from "../errors.js";
import type { DiscoveredResourceRecord } from "../models.js";
import type {
  DiscoveredResourceRepository,
  DiscoverySyncItem,
  PluginConnectionRepository,
} from "../repositories/types.js";
import { assertNoPlaintextSecrets } from "../secrets.js";

export class ResourceDiscoveryService {
  constructor(
    private readonly connections: PluginConnectionRepository,
    private readonly discoveredResources: DiscoveredResourceRepository,
  ) {}

  async sync(input: {
    connectionId: string;
    installedPluginId: string;
    pluginId: string;
    items: DiscoverySyncItem[];
  }): Promise<DiscoveredResourceRecord[]> {
    const connection = await this.connections.getById(input.connectionId);
    if (!connection) {
      throw new PluginConnectionNotFoundError(input.connectionId);
    }
    if (
      connection.status === "disconnected" ||
      connection.status === "revoked"
    ) {
      throw new PluginDomainError(
        "Cannot discover resources on a disconnected connection",
      );
    }

    for (const item of input.items) {
      assertNoPlaintextSecrets(item.metadata, "discovery.metadata");
    }

    const discoveredAt = new Date().toISOString();
    const results = await this.discoveredResources.syncDiscovery({
      pluginId: input.pluginId,
      installedPluginId: input.installedPluginId,
      connectionId: input.connectionId,
      discoveredAt,
      items: input.items,
    });

    const updatedConnection = {
      ...connection,
      lastSuccessfulSyncAt: discoveredAt,
      updatedAt: discoveredAt,
      lastErrorCode: undefined,
    };
    await this.connections.save(updatedConnection);
    return results;
  }

  async listByConnectionId(
    connectionId: string,
  ): Promise<DiscoveredResourceRecord[]> {
    return this.discoveredResources.listByConnectionId(connectionId);
  }
}
