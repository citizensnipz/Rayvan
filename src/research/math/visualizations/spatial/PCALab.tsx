import { useMemo, useState } from "react";
import { VisualizationShell } from "../../components/VisualizationShell";
import { ParameterControl } from "../../components/ParameterControls";
import { useMathStep } from "../../components/AnimatedMathStep";
import { GeometryView } from "../../renderers/GeometryView";
import {
  cloud,
  covariance,
  f3,
  format,
  mix,
  orthogonalStep,
  mv,
  plus,
  times,
  type V3,
} from "../../core/geometry";
export function PCALab() {
  const [noise, setNoise] = useState(0.2),
    [kind, setKind] = useState<"line" | "sheet" | "volume">("sheet");
  const points = useMemo(
    () => cloud(kind, noise).map((p) => plus(p, [1, 0.5, 1])),
    [kind, noise],
  );
  const analysis = useMemo(() => covariance(points), [points]);
  const { mean, centered, values, vectors } = analysis;
  const { t, timeline } = useMathStep(
    81,
    "PCA: raw → centred → components → basis → reduction",
    90,
  );
  const stage = Math.min(3, Math.floor(t * 4)),
    phase = Math.min(1, t * 4 - stage);
  const coordinates = centered.map((p) => mv(vectors, p));
  const display = points.map((p, i) =>
    stage === 0
      ? mix(p, centered[i], phase)
      : stage === 1
        ? centered[i]
        : stage === 2
          ? mv(orthogonalStep(vectors, phase), centered[i])
          : mix(
              coordinates[i],
              [coordinates[i][0], coordinates[i][1], 0],
              phase,
            ),
  );
  const total = values.reduce((a, b) => a + b, 0),
    retained = total ? (values[0] + values[1]) / total : 1;
  return (
    <VisualizationShell
      title="PCA: 3D → 2D"
      context="ESTABLISHED MATH / TOY DATA PROJECTION"
      subtitle="A computed PCA decomposition—not a hand-picked camera angle."
      equation={`C = XᵀX/(n−1); C vᵢ = λᵢvᵢ; scores = (x − mean)·vᵢ; retained variance = ${format(100 * retained)}%`}
      controls={
        <>
          <label>
            <span>Cloud shape</span>
            <select
              value={kind}
              onChange={(e) => setKind(e.target.value as typeof kind)}
            >
              <option value="line">Line</option>
              <option value="sheet">Sheet</option>
              <option value="volume">Volume</option>
            </select>
          </label>
          <ParameterControl
            label="Off-structure noise"
            min={0}
            max={1}
            value={noise}
            onChange={setNoise}
          />
          <p className="math-help">
            Stage {stage + 1}:{" "}
            {
              [
                "Mean centring",
                "Covariance principal directions",
                "Change to PCA basis",
                "Discard third component",
              ][stage]
            }
            . The basis stage rotates; if needed it also flips an axis to match
            PCA orientation.
          </p>
        </>
      }
      values={[
        ["Mean", f3(mean)],
        ["Covariance eigenvalues", values.map(format).join(" / ")],
        ["Retained variance", `${format(100 * retained)}%`],
        ["Discarded variance", format(values[2])],
        ["PC1", f3(vectors[0])],
        ["PC2", f3(vectors[1])],
        ["PC3", f3(vectors[2])],
      ]}
      interpretation="Centring subtracts the mean. Principal directions diagonalize sample covariance. The final flattening drops PC3: separated points may overlap, distances can shrink, and nearest neighbours can change. Low discarded variance does not prove that routing-relevant information survived."
      meaning="The same data contract can later carry projected EMC latent samples. This example computes PCA on synthetic 3D data only; it does not extract real model states."
      timeline={timeline}
    >
      <GeometryView
        caption="Projection demonstration: original dimensionality 3; final retained dimensionality 2. PCA preserves maximum variance for a linear 2D subspace, not every geometric relationship."
        scene={{
          clouds: [{ points: display, color: "#38c6cc" }],
          arrows:
            stage === 1
              ? vectors.map((v, i) => ({
                  to: times(v, Math.sqrt(Math.max(0, values[i]))),
                  color: ["#d8ff75", "#b299ff", "#ef7b86"][i],
                  label: `PC${i + 1}`,
                }))
              : [],
          paths:
            stage === 3
              ? coordinates
                  .filter((_, i) => i % 8 === 0)
                  .map((p) => ({
                    points: [p, [p[0], p[1], 0] as V3],
                    color: "#ef7b86",
                    dashed: true,
                  }))
              : [],
        }}
      />
      <div className="math-coordinate-table">
        <table>
          <thead>
            <tr>
              <th>Covariance row</th>
              <th>Values</th>
            </tr>
          </thead>
          <tbody>
            {analysis.matrix.map((r, i) => (
              <tr key={i}>
                <td>{i + 1}</td>
                <td>{f3(r)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </VisualizationShell>
  );
}
