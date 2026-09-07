import { useMemo, useState } from "react";
import { VisualizationShell } from "../../components/VisualizationShell";
import { ParameterControl } from "../../components/ParameterControls";
import { useMathStep } from "../../components/AnimatedMathStep";
import { GeometryView } from "../../renderers/GeometryView";
import { cosine3, f3, format, length, minus } from "../../core/geometry";
import {
  embeddingSamples,
  nearestSamples,
  normalizeEmbedding,
  pcaProjector,
  varianceSpectrum,
} from "../../core/representations";

const colours = ["#38c6cc", "#d8ff75", "#b99bff"];
export function EmbeddingCloudLab() {
  const [spread, setSpread] = useState(0.3),
    [separation, setSeparation] = useState(1);
  const [selected, setSelected] = useState(0),
    [compared, setCompared] = useState(40);
  const [projection, setProjection] = useState("raw");
  const { t, setIndex, timeline } = useMathStep(
    41,
    "Normalize embeddings · raw → unit length",
    90,
  );
  const samples = useMemo(
    () => embeddingSamples(separation, spread),
    [separation, spread],
  );
  const raw = samples.map((s) => s.coordinates),
    points = raw.map((p) => normalizeEmbedding(p, t));
  const pca = pcaProjector(points, 2);
  const displayed = projection === "pca" ? points.map(pca.project) : points;
  const a = points[selected],
    b = points[compared],
    similarity = cosine3(a, b);
  const euclidean = nearestSamples(points, a, 1, "euclidean", selected)[0];
  const angular = nearestSamples(points, a, 1, "cosine", selected)[0];
  return (
    <VisualizationShell
      title="Embedding point clouds / projection"
      context="SYNTHETIC 3D EMBEDDINGS / CLUSTER EXPLORATION"
      subtitle="Explore cluster separation, vector length and direction; then discard a PCA component."
      equation="x̂ = x / ‖x‖; cos θ = a·b/(‖a‖‖b‖); ‖â − b̂‖² = 2(1 − cos θ)"
      controls={
        <>
          <ParameterControl
            label="Cluster separation"
            value={separation}
            onChange={setSeparation}
            min={0.2}
            max={1.2}
          />
          <ParameterControl
            label="Cluster spread"
            value={spread}
            onChange={setSpread}
            min={0}
            max={0.8}
            step={0.05}
          />
          <ParameterControl
            label="Reference sample"
            value={selected}
            onChange={setSelected}
            min={0}
            max={119}
            step={1}
          />
          <ParameterControl
            label="Compare sample"
            value={compared}
            onChange={setCompared}
            min={0}
            max={119}
            step={1}
          />
          <label>
            <span>Embedding display</span>
            <select
              value={projection}
              onChange={(e) => setProjection(e.target.value)}
            >
              <option value="raw">Original 3D coordinates</option>
              <option value="pca">PCA → 2D (discard PC3)</option>
            </select>
          </label>
          <button
            onClick={() => {
              setSpread(0.3);
              setSeparation(1);
              setSelected(0);
              setCompared(40);
              setProjection("raw");
              setIndex(0);
            }}
          >
            Reset embeddings
          </button>
        </>
      }
      values={[
        [
          "Reference / compared IDs",
          `${samples[selected].id} / ${samples[compared].id}`,
        ],
        ["Reference coordinates (pre-projection)", f3(a)],
        ["Vector lengths", `${format(length(a))} / ${format(length(b))}`],
        ["Euclidean distance (pre-projection)", format(length(minus(a, b)))],
        ["Cosine similarity (pre-projection)", format(similarity)],
        [
          "Angle",
          similarity === null
            ? "Undefined"
            : `${format((Math.acos(Math.max(-1, Math.min(1, similarity))) * 180) / Math.PI)}°`,
        ],
        [
          "Euclidean / cosine nearest IDs",
          `${euclidean ? samples[euclidean.index].id : "Undefined"} / ${angular ? samples[angular.index].id : "Undefined"}`,
        ],
        ["Normalization progress", `${Math.round(t * 100)}%`],
        [
          "Displayed pair distance",
          format(length(minus(displayed[selected], displayed[compared]))),
        ],
        [
          "Discarded variance",
          projection === "pca"
            ? `${format(varianceSpectrum(pca.stats.values).shares[2] * 100)}%`
            : "None",
        ],
      ]}
      interpretation={`Cyan and lime clusters point in similar directions but have different lengths. Normalization removes this radial separation while preserving nonzero-vector cosine similarity. ${t === 1 ? "At unit length, Euclidean and cosine rankings agree (apart from ties)." : "Scrub toward unit length to watch the two clusters converge."} Purple represents a different direction. Colours denote synthetic groups, not inferred semantics.`}
      meaning="Embedding magnitude and angular relationships encode different information. PCA can hide separation; the numeric comparisons remain in the full toy space so projection distortion is visible. This is not evidence of actual EMC cluster structure."
      timeline={timeline}
    >
      <GeometryView
        caption={`120 synthetic 3D samples. ${projection === "pca" ? "Projected onto PC1/PC2; PC3 discarded. Arrows originate at the projected original origin, not the centred mean." : "Original coordinates; the 2D camera view drops Z."} Cyan/lime/purple = groups 1/2/3. White segment = selected pair.`}
        scene={{
          clouds: colours.map((color, group) => ({
            color,
            points: displayed.filter((_, i) => samples[i].group === group),
          })),
          points: [
            {
              position: displayed[selected],
              color: "#ffffff",
              label: `a · ${samples[selected].id}`,
            },
            {
              position: displayed[compared],
              color: "#ffb86b",
              label: `b · ${samples[compared].id}`,
            },
          ],
          arrows: [selected, compared].map((i, j) => ({
            from: projection === "pca" ? pca.project([0, 0, 0]) : [0, 0, 0],
            to: displayed[i],
            color: j ? "#ffb86b" : "#ffffff",
          })),
          paths: [
            {
              points: [displayed[selected], displayed[compared]],
              color: "#ffffff",
            },
          ],
        }}
      />
    </VisualizationShell>
  );
}
