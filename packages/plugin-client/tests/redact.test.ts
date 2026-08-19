import { describe, expect, it } from "vitest";

import { redactPluginLogText } from "../src/redact.js";

describe("redactPluginLogText", () => {
  it("masks common credential shapes", () => {
    expect(redactPluginLogText("token=gho_abcdefghijklmnop")).toContain("***");
    const bearer = redactPluginLogText("Authorization: Bearer abc.def.ghi");
    expect(bearer).toContain("Authorization:");
    expect(bearer).not.toContain("abc.def.ghi");
    expect(bearer).toContain("***");
    expect(redactPluginLogText("using github_pat_ABCDEFGHIJKLMNOP")).toContain(
      "***",
    );
    expect(redactPluginLogText("plain log line")).toBe("plain log line");
  });
});
