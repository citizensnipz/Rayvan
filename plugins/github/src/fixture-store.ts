import { scanWorkflowReferences, type WorkflowReferenceScan } from "./workflows.js";

export interface FixtureRepository {
  fullName: string;
  name: string;
  owner: string;
  private: boolean;
  defaultBranch: string;
}

export interface FixtureVariable {
  name: string;
  value: string;
}

export interface FixtureState {
  login: string;
  repositories: FixtureRepository[];
  /** keyed by fullName */
  variables: Record<string, FixtureVariable[]>;
  /** keyed by fullName */
  workflows: Record<string, Array<{ path: string; content: string }>>;
  workflowRefs: Record<string, WorkflowReferenceScan>;
}

function buildDefaultState(): FixtureState {
  const workflows = {
    "rayvan-fixture/demo-app": [
      {
        path: ".github/workflows/ci.yml",
        content: `
name: ci
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: echo "node \${{ vars.NODE_VERSION }}"
      - run: echo "needs \${{ vars.MISSING_VAR }}"
      - run: echo "token \${{ secrets.DEPLOY_TOKEN }}"
`,
      },
    ],
  } satisfies FixtureState["workflows"];

  const workflowRefs: Record<string, WorkflowReferenceScan> = {};
  for (const [fullName, files] of Object.entries(workflows)) {
    workflowRefs[fullName] = scanWorkflowReferences(files);
  }

  return {
    login: "rayvan-fixture",
    repositories: [
      {
        fullName: "rayvan-fixture/demo-app",
        name: "demo-app",
        owner: "rayvan-fixture",
        private: false,
        defaultBranch: "main",
      },
    ],
    variables: {
      "rayvan-fixture/demo-app": [
        { name: "NODE_VERSION", value: "20" },
        { name: "UNUSED_FLAG", value: "1" },
      ],
    },
    workflows,
    workflowRefs,
  };
}

let state: FixtureState = buildDefaultState();

export function resetGithubFixtureStore(): void {
  state = buildDefaultState();
}

export function getGithubFixtureStore(): FixtureState {
  return state;
}

export function upsertFixtureVariable(
  fullName: string,
  name: string,
  value: string,
): void {
  const list = state.variables[fullName] ?? [];
  const existing = list.find((item) => item.name === name);
  if (existing) {
    existing.value = value;
  } else {
    list.push({ name, value });
  }
  state.variables[fullName] = list;
}
