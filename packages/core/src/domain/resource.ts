export interface Resource {
  id: string;
  integrationId: string;
  kind: string;
  name: string;
  metadata?: Record<string, string>;
}
