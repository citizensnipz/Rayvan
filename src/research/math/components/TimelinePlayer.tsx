import { useEffect, useState } from "react";
import type { Frame } from "../types";
export function TimelinePlayer<T>({
  frames,
  index,
  onSelect,
  onAppend,
  onRemove,
  onGenerate,
  allowPlayback = true,
  label = "MANUAL SEQUENCE",
  intervalMs = 750,
}: {
  frames: readonly Frame<T>[];
  index: number;
  onSelect: (index: number) => void;
  onAppend?: () => void;
  onRemove?: () => void;
  onGenerate?: () => void;
  allowPlayback?: boolean;
  label?: string;
  intervalMs?: number;
}) {
  const [playing, setPlaying] = useState(false);
  const last = frames.length - 1;
  const [reducedMotion, setReducedMotion] = useState(false);
  useEffect(() => {
    const media = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    if (!media) return;
    setReducedMotion(media.matches);
    const update = () => setReducedMotion(media.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  useEffect(() => {
    const stop = () => {
      if (document.hidden) setPlaying(false);
    };
    document.addEventListener("visibilitychange", stop);
    return () => document.removeEventListener("visibilitychange", stop);
  }, []);
  useEffect(() => {
    if (!playing || !allowPlayback || last < 1) return;
    if (index >= last) {
      setPlaying(false);
      return;
    }
    const timer = window.setTimeout(
      () => onSelect(index + 1),
      reducedMotion ? Math.max(750, intervalMs) : intervalMs,
    );
    return () => window.clearTimeout(timer);
  }, [
    playing,
    allowPlayback,
    index,
    last,
    onSelect,
    intervalMs,
    reducedMotion,
  ]);
  const select = (n: number) => {
    setPlaying(false);
    onSelect(n);
  };
  return (
    <section className="math-timeline panel" aria-label="Timeline">
      <div className="math-timeline-top">
        <span className="eyebrow">{label}</span>
        <span>
          {frames.length
            ? `Step ${frames[index]?.step ?? index} · frame ${index + 1} / ${frames.length}`
            : "No frames"}
        </span>
      </div>
      <div className="math-playback">
        <button
          aria-label="Previous frame"
          disabled={index <= 0}
          onClick={() => select(index - 1)}
        >
          ←
        </button>
        {allowPlayback && (
          <button
            disabled={last < 1}
            onClick={() => {
              if (!playing && index >= last) onSelect(0);
              setPlaying(!playing);
            }}
          >
            {playing ? "Pause" : "Play"}
          </button>
        )}
        <input
          aria-label="Timeline frame"
          type="range"
          min={0}
          max={Math.max(0, last)}
          step={1}
          value={index}
          disabled={last < 1}
          onChange={(e) => select(e.target.valueAsNumber)}
        />
        <button
          aria-label="Next frame"
          disabled={index >= last}
          onClick={() => select(index + 1)}
        >
          →
        </button>
      </div>
      <div className="math-sequence-actions">
        <button disabled={!frames.length} onClick={() => select(0)}>
          Reset playback
        </button>
        {onAppend && (
          <button
            disabled={frames.length >= 24}
            onClick={() => {
              setPlaying(false);
              onAppend();
            }}
          >
            Add step
          </button>
        )}
        {onRemove && (
          <button
            disabled={frames.length <= 1}
            onClick={() => {
              setPlaying(false);
              onRemove();
            }}
          >
            Remove step
          </button>
        )}
        {onGenerate && (
          <button
            onClick={() => {
              setPlaying(false);
              onGenerate();
            }}
          >
            Load example sequence
          </button>
        )}
        <small>
          {onAppend
            ? "Edit the selected step with the controls above. Up to 24 steps."
            : "Every frame is inspectable while paused. Scrub or step to compare."}
          {reducedMotion ? " Reduced motion: slow, discrete playback." : ""}
        </small>
      </div>
    </section>
  );
}
export function useSequence<T>(initial: T, examples: () => T[]) {
  const [states, setStates] = useState<T[]>([initial]);
  const [index, setIndex] = useState(0);
  const update = (state: T) =>
    setStates((current) =>
      current.map((value, i) => (i === index ? state : value)),
    );
  const frames = states.map((state, step) => ({ state, step }));
  const player = (
    <TimelinePlayer
      frames={frames}
      index={index}
      onSelect={setIndex}
      onAppend={() => {
        setStates([...states, structuredClone(states[index])]);
        setIndex(states.length);
      }}
      onRemove={() => {
        setStates(states.filter((_, i) => i !== index));
        setIndex(Math.max(0, index - 1));
      }}
      onGenerate={() => {
        setStates(examples());
        setIndex(0);
      }}
    />
  );
  return { state: states[index], update, player };
}
