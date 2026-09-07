import { useMemo, useState } from "react";
import { VisualizationShell } from "../../components/VisualizationShell";
import { ParameterControl } from "../../components/ParameterControls";
import { useMathStep } from "../../components/AnimatedMathStep";
import { GeometryView } from "../../renderers/GeometryView";
import { covariance, format, plus, times } from "../../core/geometry";
import {
  dimensionSamples,
  varianceSpectrum,
  type IntrinsicShape,
} from "../../core/representations";

const colours = ["#38c6cc", "#d8ff75", "#b99bff"];
export function IntrinsicDimensionLab() {
  const [kind, setKind] = useState<IntrinsicShape>("sheet"),
    [noise, setNoise] = useState(0);
  const { t, setIndex, timeline } = useMathStep(
    41,
    "Unfold structure · line → selected shape",
    90,
    40,
  );
  const points = useMemo(
    () => dimensionSamples(kind, noise, t),
    [kind, noise, t],
  );
  const stats = useMemo(() => covariance(points), [points]),
    spectrum = varianceSpectrum(stats.values);
  const ideal = kind === "line" || t === 0 ? 1 : kind === "volume" ? 3 : 2;
  return (
    <VisualizationShell
      title="Intrinsic dimension intuition"
      context="TOY GENERATING STRUCTURE / NOT AN INTRINSIC-DIMENSION ESTIMATOR"
      subtitle="A cloud can occupy three axes while requiring only one or two independent parameters."
      equation="x = g(u₁,…,u_d) ∈ ℝ³; C = XᵀX/(n−1); d_eff = (Σλᵢ)² / Σλᵢ²"
      controls={
        <>
          <label>
            <span>Structure</span>
            <select
              value={kind}
              onChange={(e) => setKind(e.target.value as IntrinsicShape)}
            >
              <option value="line">Line · one parameter</option>
              <option value="sheet">Flat sheet · two parameters</option>
              <option value="curved">Curved sheet · two parameters</option>
              <option value="volume">Volume · three parameters</option>
            </select>
          </label>
          {kind !== "volume" && (
            <ParameterControl
              label="Off-structure thickness"
              min={0}
              max={0.5}
              step={0.025}
              value={noise}
              onChange={setNoise}
            />
          )}
          <button
            onClick={() => {
              setKind("sheet");
              setNoise(0);
              setIndex(40);
            }}
          >
            Reset dimension example
          </button>
        </>
      }
      values={[
        ["Ambient dimensions", 3],
        ["Noiseless generating dimensions", ideal],
        ["Global linear effective dimension", format(spectrum.participation)],
        ["PCs for 95% of variance", spectrum.dimensions95],
        [
          "Covariance eigenvalues",
          stats.values.map((v) => format(Math.max(0, v))).join(" / "),
        ],
        ["Unfolding", `${Math.round(t * 100)}%`],
        [
          "Off-structure thickness",
          kind === "volume" ? "Not applicable to a full volume" : format(noise),
        ],
      ]}
      interpretation={`${kind === "curved" ? "The curved sheet still needs only two parameters, but its bending produces variance in three global linear directions. A global PCA spectrum is not the same thing as intrinsic dimension." : `The ideal ${kind} uses ${kind === "line" ? "one" : kind === "sheet" ? "two" : "three"} independent parameters. The arrows show covariance directions, with lengths proportional to standard deviation.`} Nonzero thickness adds off-structure variation; finite noisy samples do not define an exact manifold dimension. The participation ratio measures how evenly variance is distributed, not a count of true latent factors.`}
      meaning="A high-dimensional representation may occupy a lower-dimensional structure. Curvature, noise and observation scale affect what a linear projection reveals; these synthetic examples do not estimate EMC's intrinsic dimension."
      timeline={kind === "line" ? null : timeline}
    >
      <GeometryView
        caption="Synthetic structures embedded in 3D. Coloured arrows = covariance axes (1 standard deviation). A 2D view drops Z and may conceal thickness."
        scene={{
          clouds: [{ points, color: "#7894ad" }],
          points: [{ position: stats.mean, label: "mean", color: "#ffffff" }],
          arrows: stats.vectors.map((v, i) => ({
            from: stats.mean,
            to: plus(
              stats.mean,
              times(v, Math.sqrt(Math.max(0, stats.values[i]))),
            ),
            color: colours[i],
            label: `PC${i + 1}`,
          })),
        }}
      />
      <div aria-label="Variance spectrum" className="math-spectrum">
        {spectrum.shares.map((share, i) => (
          <label key={i}>
            <span style={{ color: colours[i] }}>
              PC{i + 1} · {(share * 100).toFixed(1)}%
            </span>
            <meter
              min={0}
              max={1}
              value={share}
              aria-label={`PC${i + 1} variance fraction`}
            />
          </label>
        ))}
      </div>
    </VisualizationShell>
  );
}
