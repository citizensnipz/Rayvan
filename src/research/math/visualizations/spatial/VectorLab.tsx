import { useState } from "react";
import { VisualizationShell } from "../../components/VisualizationShell";
import { Vector3Controls } from "../../components/SpatialControls";
import { ParameterControl } from "../../components/ParameterControls";
import { useMathStep } from "../../components/AnimatedMathStep";
import { GeometryView } from "../../renderers/GeometryView";
import {
  cosine3,
  f3,
  format,
  inner,
  length,
  minus,
  mix,
  plus,
  times,
  unit,
  type V3,
} from "../../core/geometry";
export type VectorMode =
  | "vectors"
  | "dot"
  | "cosine"
  | "projection"
  | "operations"
  | "basis"
  | "gram"
  | "gradient-similarity";
export function VectorLab({ mode = "operations" }: { mode?: VectorMode }) {
  const [u, setU] = useState<V3>([2, 1, 0.5]),
    [v, setV] = useState<V3>([1, 2, 1]),
    [scalar, setScalar] = useState(1),
    [operation, setOperation] = useState("addition");
  const { t, timeline } = useMathStep(41, "Vector transformation", 70, 40),
    c = cosine3(u, v),
    projection = length(v) > 1e-12 ? times(v, inner(u, v) / inner(v, v)) : null;
  const q1 = unit(u),
    remainder = q1 ? minus(v, times(q1, inner(v, q1))) : null,
    q2 = remainder ? unit(remainder) : null;
  const result: V3 =
    mode === "projection"
      ? (projection ?? [0, 0, 0])
      : mode === "gram"
        ? (q2 ?? [0, 0, 0])
        : mode === "basis"
          ? plus(times(u, scalar), v)
          : operation === "subtraction"
            ? minus(u, v)
            : operation === "scalar"
              ? times(u, scalar)
              : plus(u, v);
  const title =
    mode === "gram"
      ? "Orthogonality / Gram–Schmidt"
      : mode === "basis"
        ? "Basis and linear combinations"
        : mode === "gradient-similarity"
          ? "Gradient-direction similarity"
          : mode === "operations"
            ? "Vector arithmetic"
            : `Spatial ${mode}`;
  const equation =
    mode === "vectors"
      ? `u = ${f3(u)}; ‖u‖ = ${format(length(u))}`
      : mode === "gram"
        ? `q₁ = u/‖u‖; r = v − (v·q₁)q₁; q₂ = r/‖r‖`
        : mode === "projection"
          ? `projᵥ(u) = (u·v)/(v·v) v = ${projection ? f3(projection) : "undefined"}`
          : mode === "basis"
            ? `w = αu + v = ${f3(result)}`
            : mode === "cosine" || mode === "gradient-similarity"
              ? `cos θ = u·v / (‖u‖ ‖v‖) = ${format(c)}`
              : mode === "dot"
                ? `u·v = ${format(inner(u, v))}`
                : operation === "subtraction"
                  ? `u − v = ${f3(result)}`
                  : operation === "scalar"
                    ? `αu = ${f3(result)}`
                    : `u + v = ${f3(result)}`;
  const validResult =
    mode === "projection"
      ? projection !== null
      : mode === "gram"
        ? q2 !== null
        : true;
  const showResult =
    ["operations", "basis", "projection", "gram"].includes(mode) && validResult;
  const shown = mix(mode === "gram" ? v : u, result, t);
  return (
    <VisualizationShell
      title={title}
      context={
        mode === "gradient-similarity"
          ? "EMC HYPOTHESIS / TOY GRADIENTS"
          : "ESTABLISHED MATH / 3D TOY"
      }
      subtitle="Coordinates, direction and magnitude remain inspectable at every animation frame."
      equation={equation}
      controls={
        <>
          <Vector3Controls label="u" value={u} onChange={setU} />
          <Vector3Controls label="v" value={v} onChange={setV} />
          {["operations", "basis"].includes(mode) && (
            <>
              {mode === "operations" && (
                <label>
                  <span>Operation</span>
                  <select
                    value={operation}
                    onChange={(e) => setOperation(e.target.value)}
                  >
                    <option value="addition">
                      Addition / linear combination
                    </option>
                    <option value="subtraction">Subtraction</option>
                    <option value="scalar">Scalar multiplication</option>
                  </select>
                </label>
              )}
              {(mode === "basis" || operation === "scalar") && (
                <ParameterControl
                  label="Scalar α"
                  value={scalar}
                  min={-3}
                  max={3}
                  onChange={setScalar}
                />
              )}
            </>
          )}
          <button
            onClick={() => {
              setU([2, 1, 0.5]);
              setV([1, 2, 1]);
              setScalar(1);
            }}
          >
            Reset vectors
          </button>
        </>
      }
      values={[
        ["u", f3(u)],
        ["v", f3(v)],
        ["‖u‖", format(length(u))],
        ["‖v‖", format(length(v))],
        ["Dot", format(inner(u, v))],
        ["Cosine", format(c)],
        ["Cosine distance", format(c === null ? null : 1 - c)],
        [
          "Angle",
          c === null
            ? "Undefined"
            : `${format((Math.acos(c) * 180) / Math.PI)}°`,
        ],
        ["Euclidean distance", format(length(minus(u, v)))],
        ...(showResult
          ? ([
              ["Target result", f3(result)],
              ["Animated result", f3(shown)],
            ] as [string, string][])
          : []),
      ]}
      interpretation={
        mode === "gram"
          ? q1 && q2
            ? "Subtracting v’s component along u leaves a perpendicular residual. Normalization makes the pair orthonormal."
            : "A zero or dependent input cannot supply a second independent basis vector; the missing direction is undefined."
          : mode === "basis"
            ? "The green vector is a weighted combination. Two vectors span at most a plane in 3D; they are not a full 3D basis."
            : mode === "gradient-similarity"
              ? "Similar directions suggest compatible infinitesimal descent directions only when vectors use the same parameter coordinates and loss convention. This is not evidence that the tasks need the same expert."
              : c === null
                ? "A zero vector has no direction: cosine and angle are undefined."
                : "Changing magnitude changes the dot product, while positive rescaling preserves cosine similarity. The animated arrow is an interpolation, not an extra mathematical operator."
      }
      meaning={
        mode === "gradient-similarity"
          ? "Exploratory: gradient signatures might describe computational need. These three coordinates stand in for a shared parameter space, not token embedding axes."
          : "These operations underlie latent updates, projections and comparisons. A 2D screen view discards depth."
      }
      timeline={
        ["operations", "basis", "projection", "gram"].includes(mode)
          ? timeline
          : null
      }
    >
      <GeometryView
        scene={{
          arrows: [
            {
              to: u,
              color: "#38c6cc",
              label: mode === "gradient-similarity" ? "∇L task A" : "u",
            },
            {
              to: v,
              color: "#b299ff",
              label: mode === "gradient-similarity" ? "∇L task B" : "v",
            },
            ...(showResult
              ? [{ to: shown, color: "#d8ff75", label: "current" }]
              : []),
            ...(mode === "gram" && q1
              ? [{ to: q1, color: "#ef7b86", label: "q₁" }]
              : []),
          ],
          points: [
            { position: u, color: "#38c6cc", onMove: setU },
            { position: v, color: "#b299ff", onMove: setV },
          ],
          paths:
            mode === "projection" && projection
              ? [{ points: [u, projection], dashed: true }]
              : [],
        }}
      />
    </VisualizationShell>
  );
}
