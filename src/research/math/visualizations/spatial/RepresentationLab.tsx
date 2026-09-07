import { useMemo, useState } from "react";
import { VisualizationShell } from "../../components/VisualizationShell";
import { ParameterControl } from "../../components/ParameterControls";
import {
  ProjectionControls,
  type ProjectionSelection,
} from "../../components/ProjectionControls";
import { GeometryView } from "../../renderers/GeometryView";
import {
  cloud,
  cosine3,
  covariance,
  f3,
  format,
  length,
  minus,
  mv,
  neighbours,
  type V3,
} from "../../core/geometry";
import { useMathStep } from "../../components/AnimatedMathStep";
export function RepresentationLab({
  mode = "intrinsic",
}: {
  mode?: "intrinsic" | "neighbours" | "embeddings";
}) {
  const [kind, setKind] = useState<"line" | "sheet" | "volume">("sheet"),
    [noise, setNoise] = useState(0),
    [selection, setSelection] = useState<ProjectionSelection>({
      method: "raw",
      axes: [0, 1, 2],
    }),
    [k, setK] = useState(8);
  const raw = useMemo(() => cloud(kind, noise), [kind, noise]),
    stats = useMemo(() => covariance(raw), [raw]);
  const points = useMemo(
    () =>
      raw.map((p, i) =>
        selection.method === "pca"
          ? mv(stats.vectors, stats.centered[i])
          : (selection.axes.map((axis) => p[axis]) as unknown as V3),
      ),
    [raw, selection, stats],
  );
  const { index, timeline } = useMathStep(raw.length, "Select a sample", 180),
    p = points[index],
    near = neighbours(
      points.filter((_, i) => i !== index),
      p,
      k,
    );
  const originalNear = neighbours(
    raw.filter((_, i) => i !== index),
    raw[index],
    k,
  );
  const agreement =
    near.filter((n) => originalNear.some((o) => o.index === n.index)).length /
    k;
  const other = points[(index + 1) % points.length],
    c = cosine3(p, other);
  return (
    <VisualizationShell
      title={
        mode === "intrinsic"
          ? "Intrinsic dimension intuition"
          : mode === "neighbours"
            ? "Local neighbourhood structure"
            : "Embedding point clouds / projection"
      }
      context="SYNTHETIC DATA / REPRESENTATION INTUITION"
      subtitle="Inspect the same samples under coordinate selection or a computed PCA basis."
      equation="Covariance C = XᵀX/(n−1); neighbours = smallest ‖x − z‖; cos θ = x·z/(‖x‖‖z‖)"
      controls={
        <>
          <label>
            <span>Structure</span>
            <select
              value={kind}
              onChange={(e) => setKind(e.target.value as typeof kind)}
            >
              <option value="line">Line in 3D · intrinsic 1</option>
              <option value="sheet">Sheet in 3D · intrinsic 2</option>
              <option value="volume">Volume · intrinsic 3</option>
            </select>
          </label>
          <ParameterControl
            label="Thickness / noise"
            min={0}
            max={1}
            value={noise}
            onChange={setNoise}
          />
          <ParameterControl
            label="Nearest neighbours k"
            min={1}
            max={20}
            step={1}
            value={k}
            onChange={setK}
          />
          <ProjectionControls value={selection} onChange={setSelection} />
        </>
      }
      values={[
        ["Ambient dimensions", 3],
        [
          "Ideal intrinsic dimensions",
          kind === "line" ? 1 : kind === "sheet" ? 2 : 3,
        ],
        ["Covariance eigenvalues", stats.values.map(format).join(" / ")],
        ["Selected sample", index],
        ["Displayed coordinates", f3(p)],
        ["Neighbour agreement with raw 3D", `${format(agreement * 100)}%`],
        ["Distance to next sample", format(length(minus(p, other)))],
        ["Cosine to next sample", format(c)],
      ]}
      interpretation="A line can occupy three coordinate axes while needing only one parameter. A noiseless sheet needs two. Adding nonzero thickness changes the mathematical support; the visible effective dimension also depends on scale. This is an illustration, not an intrinsic-dimension estimator. Repeated raw axes intentionally demonstrate information loss."
      meaning="Real latent embeddings may have useful low-dimensional structure, but a projection can merge distant states and alter nearest neighbours. All displayed samples are synthetic, not actual tokens."
      timeline={timeline}
    >
      <GeometryView
        caption={`Toy samples: original dimensionality 3. Display: ${selection.method === "pca" ? "all 3 PCA components (no reduction)" : "raw axes " + selection.axes.map((a) => a + 1).join("/")}. The 2D view drops the third displayed coordinate.`}
        scene={{
          clouds: [
            { points, color: "#536b84" },
            { points: near.map((n) => n.point), color: "#38c6cc" },
          ],
          points: [{ position: p, color: "#d8ff75", label: `sample ${index}` }],
          paths: near.map((n) => ({ points: [p, n.point], color: "#38c6cc" })),
        }}
      />
    </VisualizationShell>
  );
}
