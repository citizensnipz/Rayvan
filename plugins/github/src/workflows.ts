/**
 * Lightweight workflow YAML scan for Actions variable / secret references.
 * Not a full YAML parser — extracts `vars.NAME` / `secrets.NAME` tokens.
 */

const VAR_REF = /\bvars\.([A-Za-z_][A-Za-z0-9_]*)\b/g;
const SECRET_REF = /\bsecrets\.([A-Za-z_][A-Za-z0-9_]*)\b/g;

export interface WorkflowReferenceScan {
  variableNames: string[];
  secretNames: string[];
  filesScanned: number;
}

export function scanWorkflowReferences(
  files: ReadonlyArray<{ path: string; content: string }>,
): WorkflowReferenceScan {
  const variables = new Set<string>();
  const secrets = new Set<string>();

  for (const file of files) {
    for (const match of file.content.matchAll(VAR_REF)) {
      variables.add(match[1]!);
    }
    for (const match of file.content.matchAll(SECRET_REF)) {
      secrets.add(match[1]!);
    }
  }

  return {
    variableNames: [...variables].sort(),
    secretNames: [...secrets].sort(),
    filesScanned: files.length,
  };
}
