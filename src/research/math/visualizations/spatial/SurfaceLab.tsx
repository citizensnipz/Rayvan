import { useMemo, useState } from "react";
import { VisualizationShell } from "../../components/VisualizationShell";
import { Vector3Controls } from "../../components/SpatialControls";
import { ParameterControl } from "../../components/ParameterControls";
import { useMathStep } from "../../components/AnimatedMathStep";
import { GeometryView } from "../../renderers/GeometryView";
import { surfaceMesh, type SceneSurface } from "../../renderers/scene";
import {
  curvatures,
  derivatives,
  manifoldPath,
  optimization,
  surface,
  type Landscape,
} from "../../core/differential";
import {
  eigen2,
  f3,
  format,
  length,
  minus,
  plus,
  times,
  unit,
  type V3,
} from "../../core/geometry";
export type SurfaceMode =
  "descent" | "manifold" | "curvature" | "hessian" | "field";
export function SurfaceLab({ mode = "descent" }: { mode?: SurfaceMode }) {
  const [kind, setKind] = useState<Landscape>(
      mode === "manifold" ? "wave" : mode === "hessian" ? "saddle" : "bowl",
    ),
    [start, setStart] = useState<V3>([2, 1, 0]),
    [end, setEnd] = useState<V3>([-2, -1, 0]),
    [rate, setRate] = useState(0.15),
    [steps, setSteps] = useState(24);
  const descent = useMemo(
    () => optimization(kind, start, rate, steps),
    [kind, start, rate, steps],
  );
  const path = useMemo(
    () => manifoldPath(kind, start, end),
    [kind, start, end],
  );
  const frames = mode === "descent" ? descent.points : path.points;
  const { index, timeline } = useMathStep(
    frames.length,
    mode === "descent" ? "Optimization steps" : "Movement along surface",
    200,
  );
  const p = frames[index],
    g = derivatives(kind, p[0], p[1]),
    normal = unit([-g.gx, -g.gy, 1])!,
    k = curvatures(kind, p[0], p[1]);
  const h = eigen2(g.xx, g.xy, g.xy, g.yy),
    mesh = useMemo(
      () => surfaceMesh((x, y) => surface(kind, x, y), 4, 36),
      [kind],
    );
  const tangent: SceneSurface = {
    vertices: [
      [-0.6, -0.6],
      [0.6, -0.6],
      [0.6, 0.6],
      [-0.6, 0.6],
    ].map(([x, y]) => plus(p, [x, y, g.gx * x + g.gy * y])),
    indices: [0, 1, 2, 0, 2, 3],
    color: "#ffd275",
    opacity: 0.5,
  };
  const gradient: V3 = [g.gx, g.gy, 0],
    field =
      mode === "field"
        ? Array.from({ length: 25 }, (_, i) => {
            const x = (i % 5) - 2,
              y = Math.floor(i / 5) - 2,
              d = derivatives(kind, x, y),
              from: V3 = [x, y, surface(kind, x, y)];
            return {
              from,
              to: plus(from, [d.gx * 0.25, d.gy * 0.25, 0]),
              color: "#ef7b86",
            };
          })
        : [];
  const localSamples: V3[] = Array.from({ length: 32 }, (_, i) => {
    const angle = i * 2.39996,
      r = 0.5 * Math.sqrt((i + 1) / 32),
      x = p[0] + r * Math.cos(angle),
      y = p[1] + r * Math.sin(angle);
    return [x, y, surface(kind, x, y)];
  });
  const titles: Record<SurfaceMode, string> = {
    descent: "Gradient descent on a loss surface",
    manifold: "Local manifold and tangent plane",
    curvature: "Surface curvature",
    hessian: "Hessian / local curvature",
    field: "Gradient vector field",
  };
  const formulas: Record<Landscape, string> = {
    bowl: "¼(x²+y²)",
    valley: "0.1x²+y²",
    saddle: "¼(x²−y²)",
    "multi-basin": "0.15(x²+y²)+0.55(cos 2x+cos 2y)",
    plane: "0",
    wave: "0.6 sin x cos y",
  };
  return (
    <VisualizationShell
      title={titles[mode]}
      context="ESTABLISHED MATH / SYNTHETIC SURFACE"
      subtitle="z = L(x,y). Yellow is the local tangent plane; cyan is the surface."
      equation={`L(x,y) = ${formulas[kind]}; ∇L = (${format(g.gx)}, ${format(g.gy)}); p_next = p − η∇L`}
      controls={
        <>
          <label>
            <span>Landscape</span>
            <select
              value={kind}
              onChange={(e) => setKind(e.target.value as Landscape)}
            >
              {(
                [
                  "bowl",
                  "valley",
                  "saddle",
                  "multi-basin",
                  "plane",
                  "wave",
                ] as Landscape[]
              ).map((name) => (
                <option key={name}>{name}</option>
              ))}
            </select>
          </label>
          <Vector3Controls
            label="Start (z ignored)"
            value={start}
            onChange={setStart}
            min={-3}
            max={3}
          />
          {mode === "descent" ? (
            <>
              <ParameterControl
                label="Learning rate"
                min={0}
                max={1.2}
                step={0.01}
                value={rate}
                onChange={setRate}
              />
              <ParameterControl
                label="Number of steps"
                min={1}
                max={80}
                step={1}
                value={steps}
                onChange={setSteps}
              />
            </>
          ) : (
            <Vector3Controls
              label="Destination (z ignored)"
              value={end}
              onChange={setEnd}
              min={-3}
              max={3}
            />
          )}
          <button
            onClick={() => {
              setStart([2, 1, 0]);
              setEnd([-2, -1, 0]);
              setRate(0.15);
            }}
          >
            Reset example
          </button>
        </>
      }
      values={[
        ["Current position", f3(p)],
        ["Loss / height", format(p[2])],
        ["Gradient magnitude", format(length(gradient))],
        [
          "Hessian rows",
          `(${format(g.xx)}, ${format(g.xy)}); (${format(g.xy)}, ${format(g.yy)})`,
        ],
        [
          "Hessian eigenvalues",
          h.pairs.map((p) => format(p.value)).join(" / "),
        ],
        ["Gaussian curvature K", format(k.gaussian)],
        ["Principal curvatures", k.principal.map(format).join(" / ")],
        [
          "Ambient endpoint distance",
          format(length(minus(path.points[0], path.points.at(-1)!))),
        ],
        ["Approx. candidate surface-path length", format(path.distance)],
      ]}
      interpretation={
        mode === "descent" || mode === "field"
          ? `The red gradient is drawn in parameter x/y directions at the current height; it is not a tangent vector in the embedded surface. The green −η∇L arrow shows the parameter update; the trajectory re-evaluates loss at each landing point. ${descent.stopped ? "Stopped before leaving the displayed domain; no clamping is disguised as descent." : "Large learning rates may increase loss or oscillate."}`
          : mode === "hessian"
            ? "Hessian eigenvectors show parameter-space directions of second derivative. Positive values curve upward; mixed signs indicate a saddle. These are not generally the surface’s principal-curvature directions."
            : `The normal and tangent plane change with position. K is positive on a bowl, negative on a saddle, and zero on a plane. The drawn surface path follows a straight line in x/y coordinates: its sampled length approximates this candidate path, not a solved shortest geodesic. Sampling chords do not certify a strict upper bound.`
      }
      meaning="EMC hypothesis: local curvature and neighbourhood geometry might describe representation dynamics. The displayed graph is a genuine 2D toy manifold in 3D, not a measured latent manifold. Surface principal curvatures and loss Hessian eigenvalues are different quantities."
      timeline={timeline}
    >
      <GeometryView
        caption="Toy graph surface in 3D. XY view discards height; use the original 2D gradient module for contours."
        scene={{
          surfaces: [mesh, tangent],
          clouds:
            mode === "descent" || mode === "field"
              ? []
              : [{ points: localSamples, color: "#b299ff" }],
          points: [
            { position: p, label: "current", color: "#d8ff75" },
            {
              position: [start[0], start[1], surface(kind, start[0], start[1])],
              label: "start",
              onMove: (v) => setStart([v[0], v[1], 0]),
            },
          ],
          arrows: [
            {
              from: p,
              to: plus(p, times(normal, 0.8)),
              label: "normal",
              color: "#38c6cc",
            },
            ...(mode === "descent" || mode === "field"
              ? [
                  {
                    from: p,
                    to: plus(p, gradient),
                    color: "#ef7b86",
                    label: "∇L",
                  },
                  {
                    from: p,
                    to: plus(p, times(gradient, -rate)),
                    color: "#d8ff75",
                    label: "−η∇L",
                  },
                ]
              : []),
            ...(mode === "hessian"
              ? h.pairs.map((e, i) => ({
                  from: p,
                  to: plus(p, e.vector),
                  color: "#b299ff",
                  label: `H direction ${i + 1}`,
                }))
              : []),
            ...field,
          ],
          paths: [
            { points: frames, color: "#d8ff75" },
            { points: frames.slice(0, index + 1), color: "#ffd275" },
            ...(mode !== "descent"
              ? [
                  {
                    points: [path.points[0], path.points.at(-1)!],
                    color: "#ef7b86",
                    dashed: true,
                  },
                ]
              : []),
          ],
        }}
      />
    </VisualizationShell>
  );
}
