import { ParameterControl } from "./ParameterControls";
import type { V3, M3 } from "../core/geometry";
export function Vector3Controls({
  label,
  value,
  onChange,
  min = -4,
  max = 4,
}: {
  label: string;
  value: V3;
  onChange: (v: V3) => void;
  min?: number;
  max?: number;
}) {
  return (
    <fieldset className="math-vector-controls">
      <legend>{label}</legend>
      {value.map((v, i) => (
        <ParameterControl
          key={i}
          label={`${label} ${["x", "y", "z"][i]}`}
          value={v}
          min={min}
          max={max}
          onChange={(n) =>
            onChange(value.map((x, j) => (j === i ? n : x)) as unknown as V3)
          }
        />
      ))}
    </fieldset>
  );
}
export function Matrix3Controls({
  value,
  onChange,
}: {
  value: M3;
  onChange: (m: M3) => void;
}) {
  return (
    <fieldset className="math-vector-controls">
      <legend>A (rows × columns)</legend>
      <div className="math-matrix3-inputs">
        {value.flatMap((r, i) =>
          r.map((x, j) => (
            <label key={`${i}${j}`}>
              <span>
                a{i + 1}
                {j + 1}
              </span>
              <input
                type="number"
                aria-label={`A row ${i + 1} column ${j + 1}`}
                value={x}
                min={-3}
                max={3}
                step={0.1}
                onChange={(e) => {
                  const n = e.target.valueAsNumber;
                  if (Number.isFinite(n))
                    onChange(
                      value.map((row, k) =>
                        row.map((v, l) =>
                          k === i && l === j ? Math.max(-3, Math.min(3, n)) : v,
                        ),
                      ) as unknown as M3,
                    );
                }}
              />
            </label>
          )),
        )}
      </div>
    </fieldset>
  );
}
