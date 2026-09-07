import { useId } from "react";
import type { Vec2 } from "../types";
export function ParameterControl({
  label,
  value,
  onChange,
  min = -5,
  max = 5,
  step = 0.1,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  const id = useId();
  const update = (n: number) => {
    if (Number.isFinite(n)) onChange(Math.max(min, Math.min(max, n)));
  };
  return (
    <div className="math-parameter">
      <label htmlFor={`${id}-number`}>{label}</label>
      <input
        id={`${id}-number`}
        type="number"
        aria-label={label}
        value={Number(value.toFixed(4))}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          if (e.target.value !== "") update(e.target.valueAsNumber);
        }}
      />
      <input
        type="range"
        aria-label={`${label} slider`}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => update(e.target.valueAsNumber)}
      />
    </div>
  );
}
export function VectorControls({
  label,
  value,
  onChange,
  min = -5,
  max = 5,
}: {
  label: string;
  value: Vec2;
  onChange: (v: Vec2) => void;
  min?: number;
  max?: number;
}) {
  return (
    <fieldset className="math-vector-controls">
      <legend>{label}</legend>
      <ParameterControl
        label={`${label} x`}
        value={value[0]}
        min={min}
        max={max}
        onChange={(x) => onChange([x, value[1]])}
      />
      <ParameterControl
        label={`${label} y`}
        value={value[1]}
        min={min}
        max={max}
        onChange={(y) => onChange([value[0], y])}
      />
    </fieldset>
  );
}
