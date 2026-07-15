export * from "./errors.js";
export * from "./validation.js";
export * from "./repository.js";
export * from "./memory-repository.js";
export * from "./service.js";

import { InMemoryEnvironmentRepository } from "./memory-repository.js";
import {
  createInMemoryConfigurationPersistence,
  type InMemoryConfigurationPersistence,
} from "../configuration/index.js";

export interface InMemoryEnvironmentPersistence {
  environments: InMemoryEnvironmentRepository;
  configuration: InMemoryConfigurationPersistence;
}

export function createInMemoryEnvironmentPersistence(): InMemoryEnvironmentPersistence {
  return {
    environments: new InMemoryEnvironmentRepository(),
    configuration: createInMemoryConfigurationPersistence(),
  };
}
