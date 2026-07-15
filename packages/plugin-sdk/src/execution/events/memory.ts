import type {
  PluginExecutionEvent,
  PluginExecutionEventSink,
} from "./types.js";

/** In-memory event sink for tests and local diagnostics. */
export class InMemoryPluginExecutionEventSink
  implements PluginExecutionEventSink
{
  readonly events: PluginExecutionEvent[] = [];

  record(event: PluginExecutionEvent): void {
    this.events.push(event);
  }

  clear(): void {
    this.events.length = 0;
  }
}
