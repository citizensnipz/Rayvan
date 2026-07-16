/**
 * Prefer daemon-backed gateways when the daemon is connected.
 * Fall back to in-memory `createDev*` gateways when:
 * - daemon is offline, or
 * - `import.meta.env.DEV` and `VITE_FORCE_DEV_GATEWAYS=true`
 */
export function preferDaemonGateways(daemonConnected: boolean): boolean {
  const forceDev =
    import.meta.env.DEV && import.meta.env.VITE_FORCE_DEV_GATEWAYS === "true";
  if (forceDev) {
    return false;
  }
  return daemonConnected;
}
