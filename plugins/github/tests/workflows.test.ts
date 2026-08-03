import { describe, expect, it } from "vitest";

import { scanWorkflowReferences } from "../src/workflows.js";

describe("workflow reference scanning", () => {
  it("extracts vars and secrets references", () => {
    const scan = scanWorkflowReferences([
      {
        path: ".github/workflows/ci.yml",
        content: "run: echo ${{ vars.NODE_VERSION }} ${{ secrets.TOKEN }}",
      },
    ]);
    expect(scan.variableNames).toEqual(["NODE_VERSION"]);
    expect(scan.secretNames).toEqual(["TOKEN"]);
  });
});
