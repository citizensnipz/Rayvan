import { useMemo, useState } from "react";
import { VisualizationShell } from "../../components/VisualizationShell";
import {
  Matrix3Controls,
  Vector3Controls,
} from "../../components/SpatialControls";
import { useMathStep } from "../../components/AnimatedMathStep";
import { GeometryView } from "../../renderers/GeometryView";
import {
  I3,
  covariance,
  format,
  length,
  minus,
  mix,
  mv,
  neighbours,
  plus,
  spherePoints,
  type M3,
  type V3,
} from "../../core/geometry";
import { shapeError } from "../../core/routing";
export const transformations: { name: string; matrix: M3; shift: V3 }[] = [
  {
    name: "Rotation-like",
    matrix: [
      [0, -1, 0],
      [1, 0, 0],
      [0, 0, 1],
    ],
    shift: [0, 0, 0],
  },
  {
    name: "Scaling-like",
    matrix: [
      [1.8, 0, 0],
      [0, 0.7, 0],
      [0, 0, 1.2],
    ],
    shift: [0, 0, 0],
  },
  {
    name: "Shear-like",
    matrix: [
      [1, 0.8, 0],
      [0, 1, 0],
      [0, 0, 1],
    ],
    shift: [0, 0, 0],
  },
  {
    name: "Contraction",
    matrix: [
      [0.5, 0, 0],
      [0, 0.5, 0],
      [0, 0, 0.5],
    ],
    shift: [0, 0, 0],
  },
  { name: "Translation", matrix: I3, shift: [1, 0, 0] },
];
export function ShapeLab({
  mode = "shape",
}: {
  mode?: "shape" | "neighbourhood";
}) {
  const [desired, setDesired] = useState<M3>(transformations[2].matrix),
    [choice, setChoice] = useState(2),
    [desiredShift, setDesiredShift] = useState<V3>([0, 0, 0]);
  const source = useMemo(
    () => spherePoints(0.8).filter((_, i) => i % 3 === 0),
    [],
  );
  const target = transformations[choice],
    output = source.map((p) =>
      plus(
        mv(mode === "shape" ? desired : target.matrix, p),
        mode === "shape" ? desiredShift : target.shift,
      ),
    );
  const { t, timeline } = useMathStep(
      41,
      "Neighbourhood transformation",
      70,
      40,
    ),
    display = source.map((p, i) => mix(p, output[i], t));
  const rankings = transformations
    .map((expert) => ({
      expert,
      error: shapeError(
        desired,
        expert.matrix,
        source,
        expert.shift,
        desiredShift,
      ),
    }))
    .sort((a, b) => a.error - b.error);
  const best = rankings[0],
    stats = covariance(display),
    initialStats = covariance(source);
  const pairwise = (points: readonly V3[]) => {
    let sum = 0,
      count = 0;
    for (let i = 0; i < points.length; i++)
      for (let j = i + 1; j < points.length; j++) {
        sum += length(minus(points[i], points[j]));
        count++;
      }
    return sum / count;
  };
  const meanDistance = pairwise(display),
    radius = Math.max(...display.map((p) => length(minus(p, stats.mean))));
  const nearest = neighbours(display, stats.mean, 8);
  return (
    <VisualizationShell
      title={
        mode === "shape"
          ? "Transformation-shape routing intuition"
          : "Neighbourhood evolution"
      }
      context="EMC HYPOTHESIS / SYNTHETIC LOCAL TRANSFORMATIONS"
      subtitle="Compare what an expert does to a neighbourhood, not just where its prototype is."
      equation={
        mode === "shape"
          ? "match errorᵢ = mean ‖(A_desired ε + b_desired) − (Aᵢ ε + bᵢ)‖²"
          : "ε_next = Aε + b; C_next = A C Aᵀ (affine case)"
      }
      controls={
        <>
          {mode === "shape" ? (
            <>
              <Matrix3Controls value={desired} onChange={setDesired} />
              <Vector3Controls
                label="Desired translation"
                value={desiredShift}
                onChange={setDesiredShift}
              />
            </>
          ) : (
            <label>
              <span>Expert transformation</span>
              <select
                value={choice}
                onChange={(e) => setChoice(Number(e.target.value))}
              >
                {transformations.map((x, i) => (
                  <option key={x.name} value={i}>
                    {x.name}
                  </option>
                ))}
              </select>
            </label>
          )}
          <div className="math-presets">
            {transformations.map((x, i) => (
              <button
                key={x.name}
                onClick={() => {
                  if (mode === "shape") {
                    setDesired(x.matrix);
                    setDesiredShift(x.shift);
                  } else setChoice(i);
                }}
              >
                {x.name}
              </button>
            ))}
          </div>
          <p className="math-help">
            Scrub from before to after. In the shape view, purple is the best
            available expert’s full output; green is the desired transformation.
          </p>
        </>
      }
      values={[
        ...(mode === "shape"
          ? ([
              ["Best matching toy expert", best.expert.name],
              ["Match error", format(best.error)],
            ] as [string, string][])
          : []),
        ["8 nearest sample indices", nearest.map((n) => n.index).join(", ")],
        ["Mean pairwise distance before", format(pairwise(source))],
        ["Mean pairwise distance now", format(meanDistance)],
        [
          "Covariance spectrum before",
          initialStats.values.map(format).join(" / "),
        ],
        ["Covariance spectrum now", stats.values.map(format).join(" / ")],
        [
          "Anisotropy λmax/λmin",
          stats.values[2] > 1e-10
            ? format(stats.values[0] / stats.values[2])
            : "Singular",
        ],
        [
          "Density in bounding ball",
          radius > 1e-10
            ? format(display.length / ((4 / 3) * Math.PI * radius ** 3))
            : "Collapsed",
        ],
      ]}
      interpretation="Rotation preserves pairwise distances; contraction increases this toy density; unequal scaling elongates covariance. Translation changes location without changing pairwise distances. The match score is sampled displacement MSE in shared coordinates, not a validated measure of expert competence."
      meaning="Exploratory EMC idea: two states might need similar kinds of changes even if their semantic embeddings differ. Real Jacobian/gradient signatures and learned shape matching are intentionally not implemented."
      timeline={timeline}
    >
      <GeometryView
        scene={{
          clouds: [
            { points: source, color: "#536b84" },
            { points: display, color: "#d8ff75" },
            ...(mode === "neighbourhood"
              ? [{ points: nearest.map((n) => n.point), color: "#38c6cc" }]
              : []),
            ...(mode === "shape"
              ? [
                  {
                    points: source.map((p) =>
                      plus(mv(best.expert.matrix, p), best.expert.shift),
                    ),
                    color: "#b299ff",
                  },
                ]
              : []),
          ],
          points: [{ position: stats.mean, label: "centre", color: "#ffffff" }],
          paths: display
            .filter((_, i) => i % 12 === 0)
            .map((p) => ({ points: [stats.mean, p], color: "#38c6cc" })),
        }}
      />
      {mode === "shape" && (
        <div className="math-coordinate-table">
          <table>
            <thead>
              <tr>
                <th>Candidate expert</th>
                <th>Displacement MSE</th>
              </tr>
            </thead>
            <tbody>
              {rankings.map((r) => (
                <tr key={r.expert.name}>
                  <td>{r.expert.name}</td>
                  <td>{format(r.error)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </VisualizationShell>
  );
}
