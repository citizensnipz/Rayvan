export function normalizePath(path: string): string {
  return path.replace(/\\/g, "/");
}

export function joinPath(...segments: string[]): string {
  return normalizePath(segments.filter(Boolean).join("/"));
}
