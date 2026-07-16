export {
  daemonRequest,
  desktopDaemon,
  getDaemonStatus,
  listenDaemonEvents,
  reconnectDaemon,
  DaemonClientError,
  type DaemonCommandError,
  type DaemonStatusSnapshot,
  type DesktopDaemon,
} from "./client.js";
export {
  DaemonConnectionProvider,
  useDaemonConnection,
  useOptionalDaemonConnection,
  type DaemonConnectionState,
} from "./DaemonConnectionContext.js";
export { preferDaemonGateways } from "./gateway-policy.js";
