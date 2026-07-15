import type {
  ConfigurationKey,
  ConfigurationKeySource,
  ConfigurationValueType,
} from "@rayvan/core";

export interface CreateConfigurationKeyInput {
  projectId: string;
  name: string;
  description?: string;
  valueType: ConfigurationValueType;
  required: boolean;
  sensitive: boolean;
  source: ConfigurationKeySource;
}

export interface UpdateConfigurationKeyInput {
  description?: string;
  valueType?: ConfigurationValueType;
  required?: boolean;
  sensitive?: boolean;
  source?: ConfigurationKeySource;
}

export interface ConfigurationKeyRepository {
  listByProjectId(projectId: string): Promise<ConfigurationKey[]>;
  getById(id: string): Promise<ConfigurationKey | null>;
  getByProjectAndName(
    projectId: string,
    name: string,
  ): Promise<ConfigurationKey | null>;
  create(input: CreateConfigurationKeyInput): Promise<ConfigurationKey>;
  update(
    id: string,
    input: UpdateConfigurationKeyInput,
  ): Promise<ConfigurationKey>;
  delete?(id: string): Promise<void>;
}
