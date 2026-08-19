/** Best-effort redaction of credential-like tokens from plugin stderr. */
export function redactPluginLogText(text: string): string {
  // Apply Bearer first so "Authorization: Bearer <jwt>" is not double-masked
  // by the generic authorization= pattern.
  return text
    .replace(/\b(Bearer\s+)([A-Za-z0-9\-._~+/]+=*)/gi, "$1***")
    .replace(/\b(gh[pousr]_[A-Za-z0-9_]{8,})\b/g, "***")
    .replace(/\b(github_pat_[A-Za-z0-9_]{8,})\b/g, "***")
    .replace(
      /\b((?:access[_-]?token|refresh[_-]?token|api[_-]?key|secret|password|authorization)\s*[=:]\s*)(\S+)/gi,
      "$1***",
    );
}
