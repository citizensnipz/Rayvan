import type { Matrix2, Vec2 } from "../types";
import { coords, determinant, fmt, matrixMeaning, transform } from "../math";
import { ParameterControl } from "../components/ParameterControls";
import { VisualizationShell } from "../components/VisualizationShell";
import { useSequence } from "../components/TimelinePlayer";
import {
  Arrow,
  colors,
  fitExtent,
  Legend,
  VectorCanvas,
} from "../components/VectorCanvas";
const points: Vec2[] = [
  [-2, -1],
  [-1, 2],
  [1, 1],
  [2, -1],
  [0, -2],
];
const presets: [string, Matrix2][] = [
  ["Identity", [1, 0, 0, 1]],
  ["Rotate 90°", [0, -1, 1, 0]],
  ["Scale", [2, 0, 0, 0.5]],
  ["Shear", [1, 1, 0, 1]],
  ["Reflect", [-1, 0, 0, 1]],
  ["Collapse", [1, 1, 0, 0]],
];
export function MatrixTransformation() {
  const {
    state: m,
    update,
    player,
  } = useSequence<Matrix2>([1, 0.5, 0, 1], () => presets.map(([, m]) => m));
  const output = points.map((p) => transform(m, p)),
    e1 = transform(m, [1, 0]),
    e2 = transform(m, [0, 1]);
  return (
    <VisualizationShell
      title="2×2 matrix transformation"
      subtitle="The columns of A tell you where the basis vectors land."
      equation={
        <>
          <span className="math-matrix">
            <span>{fmt(m[0])}</span>
            <span>{fmt(m[1])}</span>
            <span>{fmt(m[2])}</span>
            <span>{fmt(m[3])}</span>
          </span>
          <span>
            {" "}
            × (x, y) = ({fmt(m[0])}x + {fmt(m[1])}y, {fmt(m[2])}x + {fmt(m[3])}
            y)
          </span>
        </>
      }
      controls={
        <>
          <div className="math-matrix-inputs">
            {["a", "b", "c", "d"].map((label, i) => (
              <ParameterControl
                key={label}
                label={label}
                min={-3}
                max={3}
                value={m[i]}
                onChange={(n) =>
                  update(
                    m.map((v, j) => (i === j ? n : v)) as unknown as Matrix2,
                  )
                }
              />
            ))}
          </div>
          <p className="math-help">
            A = [a b; c d]. Cyan is the original space; green is A applied to
            that space.
          </p>
          <div className="math-presets">
            {presets.map(([label, value]) => (
              <button key={label} onClick={() => update(value)}>
                {label}
              </button>
            ))}
          </div>
        </>
      }
      values={[
        ["det(A) = ad − bc", fmt(determinant(m))],
        ["A e₁ = (a, c)", coords(e1)],
        ["A e₂ = (b, d)", coords(e2)],
      ]}
      interpretation={matrixMeaning(m)}
      meaning="A learned linear layer remaps representations using this same weighted-coordinate operation, usually in many more dimensions. A singular map discards at least one input direction."
      timeline={player}
    >
      <Legend
        items={[
          ["Original grid / points", colors.original],
          ["Transformed grid / points", colors.result],
        ]}
      />
      <VectorCanvas
        title="Original and matrix-transformed basis, grid and point cloud"
        extent={fitExtent([...points, ...output, e1, e2])}
      >
        {(map) => (
          <>
            {Array.from({ length: 7 }, (_, i) => i - 3).flatMap((n) =>
              (
                [
                  [
                    [-3, n],
                    [3, n],
                  ],
                  [
                    [n, -3],
                    [n, 3],
                  ],
                ] as [Vec2, Vec2][]
              ).map(([p, q], axis) => {
                const a = transform(m, p),
                  b = transform(m, q);
                return (
                  <line
                    key={`${n}-${axis}`}
                    x1={map.x(a[0])}
                    y1={map.y(a[1])}
                    x2={map.x(b[0])}
                    y2={map.y(b[1])}
                    stroke={colors.result}
                    opacity=".2"
                  />
                );
              }),
            )}
            <polygon
              points={([[0, 0], e1, transform(m, [1, 1]), e2] as Vec2[])
                .map((p) => `${map.x(p[0])},${map.y(p[1])}`)
                .join(" ")}
              fill={colors.result}
              opacity=".12"
            />
            {points.map((p, i) => (
              <g key={i}>
                <line
                  x1={map.x(p[0])}
                  y1={map.y(p[1])}
                  x2={map.x(output[i][0])}
                  y2={map.y(output[i][1])}
                  stroke={colors.muted}
                  opacity=".45"
                  strokeDasharray="3 4"
                />
                <circle
                  cx={map.x(p[0])}
                  cy={map.y(p[1])}
                  r="5"
                  fill="none"
                  stroke={colors.original}
                />
                <circle
                  cx={map.x(output[i][0])}
                  cy={map.y(output[i][1])}
                  r="3.5"
                  fill={colors.result}
                />
                <text
                  x={map.x(output[i][0]) + 7}
                  y={map.y(output[i][1]) + 14}
                  className="math-tick"
                >
                  P{i + 1}′
                </text>
              </g>
            ))}
            <Arrow
              map={map}
              to={[1, 0]}
              color={colors.original}
              label="e₁"
              labelOffset={[8, 17]}
              dashed
            />
            <Arrow
              map={map}
              to={[0, 1]}
              color={colors.original}
              label="e₂"
              labelOffset={[8, 17]}
              dashed
            />
            <Arrow map={map} to={e1} color={colors.result} label="Ae₁" />
            <Arrow map={map} to={e2} color={colors.result} label="Ae₂" />
          </>
        )}
      </VectorCanvas>
      <div className="math-coordinate-table">
        <table>
          <thead>
            <tr>
              <th>Point</th>
              <th>Original (x, y)</th>
              <th>Transformed A(x, y)</th>
            </tr>
          </thead>
          <tbody>
            {points.map((p, i) => (
              <tr key={i}>
                <td>P{i + 1}</td>
                <td>{coords(p)}</td>
                <td>{coords(output[i])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </VisualizationShell>
  );
}
