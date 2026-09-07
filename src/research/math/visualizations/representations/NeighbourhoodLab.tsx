import { useMemo, useState } from "react";
import { VisualizationShell } from "../../components/VisualizationShell";
import { ParameterControl } from "../../components/ParameterControls";
import { Vector3Controls } from "../../components/SpatialControls";
import { GeometryView } from "../../renderers/GeometryView";
import { cloud, covariance, f3, format, type V3 } from "../../core/geometry";
import {
  nearestSamples,
  pcaProjector,
  varianceSpectrum,
  type NeighbourMetric,
} from "../../core/representations";

export function NeighbourhoodLab() {
  const [query, setQuery] = useState<V3>([0.4, 0.3, 0.6]),
    [k, setK] = useState(8);
  const [metric, setMetric] = useState<NeighbourMetric>("euclidean"),
    [projection, setProjection] = useState("raw");
  const raw = useMemo(() => cloud("volume"), []);
  const pca = useMemo(() => pcaProjector(raw, 2), [raw]);
  const points = projection === "pca" ? raw.map(pca.project) : raw;
  const displayedQuery = projection === "pca" ? pca.project(query) : query;
  const original = nearestSamples(raw, query, k, metric);
  const near = nearestSamples(points, displayedQuery, k, metric);
  const originalIDs = new Set(original.map((n) => n.index)),
    displayedIDs = new Set(near.map((n) => n.index));
  const overlap = near.filter((n) => originalIDs.has(n.index)).length;
  const local = covariance(near.map((n) => n.point)),
    spectrum = varianceSpectrum(local.values);
  const radius = near.at(-1)?.distance ?? 0;
  const density =
    metric === "euclidean" && radius > 1e-12
      ? near.length /
        (projection === "pca"
          ? Math.PI * radius ** 2
          : (4 / 3) * Math.PI * radius ** 3)
      : null;
  return (
    <VisualizationShell
      title="Local neighbourhood structure"
      context="TOY QUERY / NEAREST-NEIGHBOUR ANALYSIS"
      subtitle="Move a query, change the metric, and watch its nearest samples enter and leave the neighbourhood."
      equation={
        metric === "euclidean"
          ? "Nₖ(q) = arg k-minᵢ ‖xᵢ − q‖₂; ρ = k / volume(B(q,rₖ))"
          : "Nₖ(q) = arg k-minᵢ [1 − (xᵢ·q)/(‖xᵢ‖‖q‖)]"
      }
      controls={
        <>
          <Vector3Controls label="Query q" value={query} onChange={setQuery} />
          <ParameterControl
            label="Nearest neighbours k"
            min={2}
            max={24}
            step={1}
            value={k}
            onChange={setK}
          />
          <label>
            <span>Neighbour metric</span>
            <select
              value={metric}
              onChange={(e) => setMetric(e.target.value as NeighbourMetric)}
            >
              <option value="euclidean">Euclidean distance</option>
              <option value="cosine">Cosine distance</option>
            </select>
          </label>
          <label>
            <span>Neighbour search space</span>
            <select
              value={projection}
              onChange={(e) => setProjection(e.target.value)}
            >
              <option value="raw">Original 3D</option>
              <option value="pca">PCA → 2D (recompute neighbours)</option>
            </select>
          </label>
          <button
            onClick={() => {
              setQuery([0.4, 0.3, 0.6]);
              setK(8);
              setMetric("euclidean");
              setProjection("raw");
            }}
          >
            Reset neighbourhood
          </button>
        </>
      }
      values={[
        ["Query (original 3D)", f3(query)],
        ["Query (search coordinates)", f3(displayedQuery)],
        [
          "Nearest sample",
          near.length
            ? `${near[0].index}: ${format(near[0].distance)}`
            : "Undefined for a zero cosine query",
        ],
        [
          "Neighbour overlap with original 3D",
          original.length ? `${overlap} / ${original.length}` : "Undefined",
        ],
        ["k-th distance", near.length ? format(radius) : "Undefined"],
        [
          "Local covariance eigenvalues",
          local.values.map((v) => format(Math.max(0, v))).join(" / "),
        ],
        ["Local linear effective dimension", format(spectrum.participation)],
        [
          projection === "pca"
            ? "Count / disc area (toy density)"
            : "Count / ball volume (toy density)",
          density === null
            ? "Not defined for this metric/query"
            : format(density),
        ],
      ]}
      interpretation={`${metric === "euclidean" ? "Euclidean neighbours are closest in position. In original 3D the translucent ball reaches the k-th neighbour; density is count divided by ball volume (or disc area in PCA)." : "Cosine neighbours share direction from the origin, even when far apart. No Euclidean influence ball is drawn. Zero-length vectors have undefined cosine and are excluded."} Cyan samples are neighbours in both spaces; orange entered after projection; purple were lost. ${projection === "pca" ? "PCA centring also changes the angular origin, so cosine changes here include both centring and component removal." : "Drag the query gizmo in 3D or edit its coordinates. Increasing k changes the scale of the local statistics."}`}
      meaning="Local structure can differ from the global cloud. A projection can change which states appear nearby, so projected neighbours should not automatically be treated as the neighbours an EMC routing rule would use. Density here is illustrative and has no boundary correction."
      timeline={null}
    >
      <GeometryView
        caption={`160 synthetic 3D samples. Search: ${projection === "pca" ? "centred PC1/PC2; PC3 discarded" : "original 3D"}. Rankings always use that search space, even in the XY camera view. Query drag is available in original 3D; numeric controls always work.`}
        scene={{
          clouds: [
            { points, color: "#536b84" },
            {
              points: near
                .filter((n) => originalIDs.has(n.index))
                .map((n) => n.point),
              color: "#38c6cc",
            },
            {
              points: near
                .filter((n) => !originalIDs.has(n.index))
                .map((n) => n.point),
              color: "#ffb86b",
            },
            {
              points: original
                .filter((n) => !displayedIDs.has(n.index))
                .map((n) => points[n.index]),
              color: "#b99bff",
            },
          ],
          points: [
            {
              position: displayedQuery,
              color: "#d8ff75",
              label: "query q",
              onMove: projection === "raw" ? setQuery : undefined,
            },
          ],
          paths: near.map((n) => ({
            points: [displayedQuery, n.point],
            color: originalIDs.has(n.index) ? "#38c6cc" : "#ffb86b",
          })),
          regions:
            metric === "euclidean" && projection === "raw"
              ? [{ center: displayedQuery, radius, color: "#38c6cc" }]
              : [],
        }}
      />
      <ol
        className="math-neighbour-ranking"
        aria-label="Ranked nearest samples"
      >
        {near.map((n) => (
          <li key={n.index}>
            <span>Sample {n.index}</span>
            <strong>{format(n.distance)}</strong>
          </li>
        ))}
      </ol>
    </VisualizationShell>
  );
}
