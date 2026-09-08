import { useState } from "react";
import { TimelinePlayer } from "./TimelinePlayer";
export function useMathStep(
  count = 41,
  label = "Transformation progress",
  intervalMs = 70,
  initialIndex = 0,
) {
  const [raw, setIndex] = useState(initialIndex);
  const index = Math.min(raw, Math.max(0, count - 1));
  const timeline = (
    <TimelinePlayer
      frames={Array.from({ length: count }, (_, step) => ({
        step,
        state: step,
      }))}
      index={index}
      onSelect={setIndex}
      label={label}
      intervalMs={intervalMs}
    />
  );
  return { index, t: count > 1 ? index / (count - 1) : 0, setIndex, timeline };
}
