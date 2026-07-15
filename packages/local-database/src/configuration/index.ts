export * from "./errors.js";
export * from "./validation.js";
export * from "./key-repository.js";
export * from "./occurrence-repository.js";
export * from "./desired-repository.js";
export * from "./applied-repository.js";
export * from "./memory-key-repository.js";
export * from "./memory-occurrence-repository.js";
export * from "./memory-desired-repository.js";
export * from "./memory-applied-repository.js";
export * from "./service.js";
export * from "./desired-state-service.js";

import { InMemoryAppliedConfigurationStateRepository } from "./memory-applied-repository.js";
import { InMemoryDesiredConfigurationValueRepository } from "./memory-desired-repository.js";
import { InMemoryConfigurationKeyRepository } from "./memory-key-repository.js";
import { InMemoryConfigurationOccurrenceRepository } from "./memory-occurrence-repository.js";
import { ConfigurationDesiredStateService } from "./desired-state-service.js";
import { ConfigurationService } from "./service.js";

export interface InMemoryConfigurationPersistence {
  keys: InMemoryConfigurationKeyRepository;
  occurrences: InMemoryConfigurationOccurrenceRepository;
  desired: InMemoryDesiredConfigurationValueRepository;
  applied: InMemoryAppliedConfigurationStateRepository;
  service: ConfigurationService;
  desiredStateService: ConfigurationDesiredStateService;
}

export function createInMemoryConfigurationPersistence(): InMemoryConfigurationPersistence {
  const keys = new InMemoryConfigurationKeyRepository();
  const occurrences = new InMemoryConfigurationOccurrenceRepository();
  const desired = new InMemoryDesiredConfigurationValueRepository();
  const applied = new InMemoryAppliedConfigurationStateRepository();
  return {
    keys,
    occurrences,
    desired,
    applied,
    service: new ConfigurationService(keys, occurrences),
    desiredStateService: new ConfigurationDesiredStateService(
      keys,
      desired,
      applied,
    ),
  };
}
