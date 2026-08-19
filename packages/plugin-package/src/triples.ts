/** Platform-specific target triples for v0.1.0 plugin packages. */
export const PLUGIN_TARGET_TRIPLES = [
  "x86_64-pc-windows-msvc",
  "aarch64-apple-darwin",
  "x86_64-apple-darwin",
  "x86_64-unknown-linux-gnu",
] as const;

export type PluginTargetTriple = (typeof PLUGIN_TARGET_TRIPLES)[number];

export function isPluginTargetTriple(
  value: string,
): value is PluginTargetTriple {
  return (PLUGIN_TARGET_TRIPLES as readonly string[]).includes(value);
}

export function detectHostTargetTriple(): PluginTargetTriple {
  const platform = process.platform;
  const arch = process.arch;
  if (platform === "win32" && arch === "x64") {
    return "x86_64-pc-windows-msvc";
  }
  if (platform === "darwin" && arch === "arm64") {
    return "aarch64-apple-darwin";
  }
  if (platform === "darwin" && arch === "x64") {
    return "x86_64-apple-darwin";
  }
  if (platform === "linux" && arch === "x64") {
    return "x86_64-unknown-linux-gnu";
  }
  throw new Error(
    `Unsupported host platform for plugin packaging: ${platform}/${arch}`,
  );
}

export function binaryFileName(binaryBase: string, triple: PluginTargetTriple): string {
  return triple.includes("windows") ? `${binaryBase}.exe` : binaryBase;
}

export function packageFileName(
  pluginId: string,
  version: string,
  triple: PluginTargetTriple,
): string {
  return `${pluginId}-${version}-${triple}.rayvan-plugin`;
}
