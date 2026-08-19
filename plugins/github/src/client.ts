import {
  getGithubFixtureStore,
  upsertFixtureVariable,
  type FixtureRepository,
  type FixtureVariable,
} from "./fixture-store.js";
import { scanWorkflowReferences } from "./workflows.js";

export interface GithubAuthCredentials {
  token: string;
  tokenType?: "bearer" | "token";
}

export interface GithubClientOptions {
  /** When true, never call the network — use in-memory fixture data. */
  fixture?: boolean;
  baseUrl?: string;
  credentials?: GithubAuthCredentials;
  fetchImpl?: typeof fetch;
}

export interface GithubRepository {
  fullName: string;
  name: string;
  owner: string;
  private: boolean;
  defaultBranch: string;
}

export interface GithubActionsVariable {
  name: string;
  value: string;
}

export interface GithubWorkflowFile {
  path: string;
  content: string;
}

export class GithubClient {
  private readonly fixture: boolean;
  private readonly baseUrl: string;
  private readonly credentials?: GithubAuthCredentials;
  private readonly fetchImpl: typeof fetch;

  constructor(options: GithubClientOptions = {}) {
    this.fixture = options.fixture === true;
    this.baseUrl = (options.baseUrl ?? "https://api.github.com").replace(
      /\/$/,
      "",
    );
    this.credentials = options.credentials;
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  async authenticate(): Promise<{ ok: boolean; login?: string; message: string }> {
    if (this.fixture) {
      const store = getGithubFixtureStore();
      return {
        ok: true,
        login: store.login,
        message: `Fixture authenticated as ${store.login}`,
      };
    }
    if (!this.credentials?.token) {
      return { ok: false, message: "Missing GitHub credential token" };
    }
    const response = await this.request("GET", "/user");
    if (!response.ok) {
      return {
        ok: false,
        message: `GitHub authentication failed (${response.status})`,
      };
    }
    const body = (await response.json()) as { login?: string };
    return {
      ok: true,
      login: body.login,
      message: body.login
        ? `Authenticated as ${body.login}`
        : "Authenticated with GitHub",
    };
  }

  async listRepositories(): Promise<GithubRepository[]> {
    if (this.fixture) {
      return getGithubFixtureStore().repositories.map(mapRepo);
    }
    const response = await this.request(
      "GET",
      "/user/repos?per_page=100&sort=updated",
    );
    if (!response.ok) {
      throw new Error(`Failed to list repositories (${response.status})`);
    }
    const body = (await response.json()) as Array<{
      full_name: string;
      name: string;
      owner: { login: string };
      private: boolean;
      default_branch: string;
    }>;
    return body.map((repo) => ({
      fullName: repo.full_name,
      name: repo.name,
      owner: repo.owner.login,
      private: repo.private,
      defaultBranch: repo.default_branch,
    }));
  }

  async listActionsVariables(
    fullName: string,
  ): Promise<GithubActionsVariable[]> {
    if (this.fixture) {
      return [...(getGithubFixtureStore().variables[fullName] ?? [])];
    }
    const [owner, repo] = splitFullName(fullName);
    const response = await this.request(
      "GET",
      `/repos/${owner}/${repo}/actions/variables?per_page=100`,
    );
    if (!response.ok) {
      throw new Error(
        `Failed to list Actions variables for ${fullName} (${response.status})`,
      );
    }
    const body = (await response.json()) as {
      variables?: Array<{ name: string; value: string }>;
    };
    return (body.variables ?? []).map((item) => ({
      name: item.name,
      value: item.value,
    }));
  }

  async upsertActionsVariable(
    fullName: string,
    name: string,
    value: string,
  ): Promise<void> {
    if (this.fixture) {
      upsertFixtureVariable(fullName, name, value);
      return;
    }
    const [owner, repo] = splitFullName(fullName);
    const existing = await this.listActionsVariables(fullName);
    const found = existing.some((item) => item.name === name);
    const path = found
      ? `/repos/${owner}/${repo}/actions/variables/${encodeURIComponent(name)}`
      : `/repos/${owner}/${repo}/actions/variables`;
    const method = found ? "PATCH" : "POST";
    const response = await this.request(method, path, {
      name,
      value,
    });
    if (!response.ok && response.status !== 204) {
      throw new Error(
        `Failed to upsert variable ${name} on ${fullName} (${response.status})`,
      );
    }
  }

  async listWorkflowFiles(fullName: string): Promise<GithubWorkflowFile[]> {
    if (this.fixture) {
      return [...(getGithubFixtureStore().workflows[fullName] ?? [])];
    }
    const [owner, repo] = splitFullName(fullName);
    const response = await this.request(
      "GET",
      `/repos/${owner}/${repo}/contents/.github/workflows`,
    );
    if (response.status === 404) {
      return [];
    }
    if (!response.ok) {
      throw new Error(
        `Failed to list workflows for ${fullName} (${response.status})`,
      );
    }
    const entries = (await response.json()) as Array<{
      type: string;
      path: string;
      download_url?: string | null;
    }>;
    const files: GithubWorkflowFile[] = [];
    for (const entry of entries) {
      if (entry.type !== "file" || !entry.download_url) continue;
      if (!/\.ya?ml$/i.test(entry.path)) continue;
      const fileResponse = await this.fetchImpl(entry.download_url);
      if (!fileResponse.ok) continue;
      files.push({
        path: entry.path,
        content: await fileResponse.text(),
      });
    }
    return files;
  }

  async scanWorkflows(fullName: string) {
    if (this.fixture) {
      const store = getGithubFixtureStore();
      return (
        store.workflowRefs[fullName] ??
        scanWorkflowReferences(store.workflows[fullName] ?? [])
      );
    }
    const files = await this.listWorkflowFiles(fullName);
    return scanWorkflowReferences(files);
  }

  private async request(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<Response> {
    if (!this.credentials?.token) {
      throw new Error("GitHub credentials are required for live API calls");
    }
    const headers: Record<string, string> = {
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      Authorization: `${this.credentials.tokenType === "token" ? "token" : "Bearer"} ${this.credentials.token}`,
      "User-Agent": "rayvan-plugin-github",
    };
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
    }
    return this.fetchImpl(`${this.baseUrl}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  }
}

function mapRepo(repo: FixtureRepository): GithubRepository {
  return { ...repo };
}

function splitFullName(fullName: string): [string, string] {
  const [owner, repo, ...rest] = fullName.split("/");
  if (!owner || !repo || rest.length > 0) {
    throw new Error(`Invalid repository full name: ${fullName}`);
  }
  return [owner, repo];
}

export type { FixtureVariable };
