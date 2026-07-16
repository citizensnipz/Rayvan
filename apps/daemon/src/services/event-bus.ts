import { randomUUID } from "node:crypto";

import type {
  DaemonEvent,
  DaemonEventType,
  RayvanActor,
} from "@rayvan/daemon-contracts";

type Listener = (event: DaemonEvent) => void;

export class EventBus {
  private readonly listeners = new Set<Listener>();

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  emit(input: {
    type: DaemonEventType;
    projectId?: string;
    actor?: RayvanActor;
    correlationId?: string;
    payload?: Record<string, unknown>;
  }): DaemonEvent {
    const event: DaemonEvent = {
      eventId: `evt_${randomUUID()}`,
      type: input.type,
      timestamp: new Date().toISOString(),
      schemaVersion: "1",
      projectId: input.projectId,
      actor: input.actor,
      correlationId: input.correlationId,
      payload: input.payload ?? {},
    };
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch {
        /* never let subscriber errors kill the bus */
      }
    }
    return event;
  }
}
