import type {
  PluginExecutionEvent,
  PluginExecutionEventSink,
} from "./types.js";

/** Event sink that discards all events. */
export class NoopPluginExecutionEventSink
  implements PluginExecutionEventSink
{
  record(_event: PluginExecutionEvent): void {
    void _event;
  }
}
