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
  det3,
  eigen2,
  f3,
  format,
  length,
  mixMatrix,
  orthogonalStep,
  mm,
  mv,
  plus,
  spherePoints,
  svd3,
  transpose,
  type M3,
  type V3,
} from "../../core/geometry";
export type MatrixMode =
  "matrix" | "eigen" | "svd" | "jacobian" | "rank" | "basis-change";
export function MatrixLab({ mode = "matrix" }: { mode?: MatrixMode }) {
  const [a, setA] = useState<M3>(
      mode === "eigen"
        ? [
            [2, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
          ]
        : [
            [1.4, 0.6, 0],
            [0, 0.7, 0.2],
            [0, 0, 1],
          ],
    ),
    [center, setCenter] = useState<V3>([0, 0, 0]);
  const { t, timeline } = useMathStep(
    61,
    mode === "svd" ? "SVD: Vᵀ → Σ → U" : "Identity → selected transformation",
    70,
    60,
  );
  const { u, s, vt } = useMemo(() => svd3(a), [a]),
    sigma: M3 = [
      [s[0], 0, 0],
      [0, s[1], 0],
      [0, 0, s[2]],
    ];
  const stage = Math.min(2, Math.floor(t * 3)),
    phase = Math.min(1, t * 3 - stage);
  // Orthogonal stages rotate continuously, with explicit reflection if determinant < 0.
  const current =
    mode === "basis-change"
      ? orthogonalStep(transpose(u), t)
      : mode === "svd"
        ? stage === 0
          ? orthogonalStep(vt, phase)
          : stage === 1
            ? mm(mixMatrix(I3, sigma, phase), vt)
            : mm(orthogonalStep(u, phase), mm(sigma, vt))
        : mixMatrix(I3, a, t);
  const source = useMemo(
    () => spherePoints(mode === "jacobian" ? 0.4 : 1),
    [mode],
  );
  const origin: V3 = mode === "basis-change" ? [0, 0, 0] : center;
  const transformed = source.map((p) => plus(origin, mv(current, p))),
    eigen = eigen2(a[0][0], a[0][1], a[1][0], a[1][1]);
  const planar =
    a[0][2] === 0 && a[1][2] === 0 && a[2][0] === 0 && a[2][1] === 0;
  const rank = s.filter((n) => n > 1e-7 * Math.max(1, s[0])).length;
  const titles: Record<MatrixMode, string> = {
    matrix: "3×3 matrix transformation",
    eigen: "Eigenvectors / eigenvalues",
    svd: "SVD / transformation spectrum",
    jacobian: "Jacobian transformation geometry",
    rank: "Rank, null space and volume",
    "basis-change": "Change of basis",
  };
  const basisCoordinates = mv(transpose(u), center);
  return (
    <VisualizationShell
      title={titles[mode]}
      context={
        mode === "jacobian"
          ? "ESTABLISHED LOCAL MATH / TOY EXPERT"
          : "ESTABLISHED MATH / 3D TOY"
      }
      subtitle="Cyan: original neighbourhood. Green: transformed neighbourhood. Coloured axes show matrix columns."
      equation={
        mode === "svd"
          ? `A = U Σ Vᵀ; Σ = diag(${s.map(format).join(", ")}); stage ${stage + 1}: ${["Vᵀ basis change", "Σ scaling", "U output basis"][stage]}`
          : mode === "jacobian"
            ? "f(x + ε) ≈ f(x) + J(x)ε; manually supplied J = A"
            : mode === "basis-change"
              ? `c = Uᵀx = ${f3(basisCoordinates)}; x = Uc (orthonormal basis U from SVD)`
              : `y = Ax; det(A) = ${format(det3(a))}`
      }
      controls={
        <>
          <Matrix3Controls value={a} onChange={setA} />
          {(mode === "jacobian" || mode === "basis-change") && (
            <Vector3Controls
              label={mode === "jacobian" ? "f(x) centre" : "x"}
              value={center}
              onChange={setCenter}
            />
          )}
          <div className="math-presets">
            <button onClick={() => setA(I3)}>Identity</button>
            <button
              onClick={() =>
                setA([
                  [0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1],
                ])
              }
            >
              Rotation
            </button>
            <button
              onClick={() =>
                setA([
                  [1, 1, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                ])
              }
            >
              Shear
            </button>
            <button
              onClick={() =>
                setA([
                  [-1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                ])
              }
            >
              Reflection
            </button>
            <button
              onClick={() =>
                setA([
                  [2, 0, 0],
                  [0, 0.5, 0],
                  [0, 0, 0],
                ])
              }
            >
              Collapse
            </button>
            <button
              onClick={() =>
                setA([
                  [2, 1, 0],
                  [1, 1, 0],
                  [0, 0, 1],
                ])
              }
            >
              Real eigenvectors
            </button>
          </div>
        </>
      }
      values={[
        ["Determinant / signed volume", format(det3(a))],
        ["Rank (relative tolerance 1e−7)", rank],
        ["Nullity", 3 - rank],
        ["Singular values", s.map(format).join(" / ")],
        ["Strongest input direction", f3(vt[0])],
        ["Weakest input direction", f3(vt[2])],
        ["Ae₁", f3(mv(a, I3[0]))],
        ["Ae₂", f3(mv(a, I3[1]))],
        ["Ae₃", f3(mv(a, I3[2]))],
        ...(mode === "eigen"
          ? ([
              [
                "XY eigenvalues",
                planar
                  ? eigen.complex
                    ? "Complex: no real XY eigenvectors"
                    : eigen.pairs.map((p) => format(p.value)).join(" / ")
                  : "Choose a block-diagonal XY example",
              ],
            ] as [string, string][])
          : []),
      ]}
      interpretation={
        mode === "svd"
          ? `Singular values describe stretch, not rotation. Input directions V are mapped onto output directions U. The weakest stretch is ${format(s[2])}. Orthogonal stages rotate continuously; a factor with negative determinant also passes through an explicit reflection. Stage endpoints are the exact factors.`
          : mode === "eigen"
            ? `An eigenvector preserves its line, not necessarily its orientation: a negative eigenvalue reverses it. Yellow highlights real XY eigenvectors only for the uncoupled XY presets. Repeated or complex eigenvalues are not invented as distinct real directions.`
            : mode === "jacobian"
              ? "The local sphere becomes an ellipsoid under this manually edited Jacobian. Its singular spectrum measures local anisotropy. This is a linear approximation, not a nonlinear expert evaluated over a large region."
              : mode === "basis-change"
                ? "Purple directions are an orthonormal basis U. The displayed coordinates Uᵀx describe the same x in that basis; they are not an extra transformation of x."
                : `Rank ${rank} means ${rank} independent output directions. A near-zero singular value exposes a lost direction; the corresponding orange input vector lies in the numerical null space.`
      }
      meaning={
        mode === "jacobian" || mode === "svd"
          ? "EMC hypothesis: local transformation shape may help characterize expert competence. This sandbox does not claim the router currently uses Jacobians or spectra."
          : "Linear layers remap coordinates. Projection and rank loss can discard information; volume scaling is the absolute determinant."
      }
      timeline={timeline}
    >
      <GeometryView
        scene={{
          clouds: [
            { points: source.map((p) => plus(origin, p)), color: "#38c6cc" },
            { points: transformed, color: "#d8ff75" },
          ],
          arrows: [
            ...I3.map((v, i) => ({
              from: origin,
              to: plus(origin, mv(current, v)),
              color: ["#ef7b86", "#d8ff75", "#38c6cc"][i],
              label: `Ae${i + 1}`,
            })),
            ...(mode === "svd"
              ? vt.map((v, i) => ({
                  to: mv(a, v),
                  color: "#b299ff",
                  label: `σ${i + 1}=${format(s[i])}`,
                }))
              : []),
            ...(mode === "eigen" && planar && !eigen.complex
              ? eigen.pairs.map((p, i) => ({
                  to: mv(current, p.vector),
                  color: "#ffd275",
                  label: `λ${i + 1}=${format(p.value)}`,
                }))
              : []),
            ...(mode === "rank"
              ? vt
                  .filter((_, i) => s[i] < 1e-7 * Math.max(1, s[0]))
                  .map((v) => ({
                    to: v,
                    color: "#ffae70",
                    label: "null direction",
                  }))
              : []),
            ...(mode === "basis-change"
              ? transpose(u).map((to, i) => ({
                  to,
                  color: "#b299ff",
                  label: `u${i + 1}`,
                }))
              : []),
          ],
          points:
            mode === "jacobian" || mode === "basis-change"
              ? [
                  {
                    position: center,
                    label: mode === "jacobian" ? "f(x)" : "original x",
                    onMove: setCenter,
                  },
                  ...(mode === "basis-change"
                    ? [
                        {
                          position: mv(current, center),
                          label: "coordinates in current basis",
                          color: "#d8ff75",
                        },
                      ]
                    : []),
                ]
              : [],
        }}
      />
      {mode === "svd" && (
        <div className="math-coordinate-table">
          <table>
            <thead>
              <tr>
                <th>Factor</th>
                <th>Rows</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["U", u],
                ["Σ", sigma],
                ["Vᵀ", vt],
              ].map(([name, matrix]) => (
                <tr key={name as string}>
                  <td>{name as string}</td>
                  <td>{(matrix as M3).map(f3).join(" ; ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </VisualizationShell>
  );
}
