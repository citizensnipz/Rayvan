import { lazy, Suspense } from "react";
import type { ConceptId } from "./types";
import type { VectorMode } from "./visualizations/spatial/VectorLab";
import type { MatrixMode } from "./visualizations/spatial/MatrixLab";
import type { SurfaceMode } from "./visualizations/spatial/SurfaceLab";
import type { RoutingMode } from "./visualizations/emc/RoutingLab";
const VectorLab = lazy(() =>
  import("./visualizations/spatial/VectorLab").then((m) => ({
    default: m.VectorLab,
  })),
);
const MatrixLab = lazy(() =>
  import("./visualizations/spatial/MatrixLab").then((m) => ({
    default: m.MatrixLab,
  })),
);
const SurfaceLab = lazy(() =>
  import("./visualizations/spatial/SurfaceLab").then((m) => ({
    default: m.SurfaceLab,
  })),
);
const PCALab = lazy(() =>
  import("./visualizations/spatial/PCALab").then((m) => ({
    default: m.PCALab,
  })),
);
const RepresentationLab = lazy(() =>
  import("./visualizations/spatial/RepresentationLab").then((m) => ({
    default: m.RepresentationLab,
  })),
);
const RoutingLab = lazy(() =>
  import("./visualizations/emc/RoutingLab").then((m) => ({
    default: m.RoutingLab,
  })),
);
const ShapeLab = lazy(() =>
  import("./visualizations/emc/ShapeLab").then((m) => ({
    default: m.ShapeLab,
  })),
);
const LatentLab = lazy(() =>
  import("./visualizations/emc/LatentLab3D").then((m) => ({
    default: m.LatentLab3D,
  })),
);
export function SpatialVisualizer({ concept }: { concept: ConceptId }) {
  let content;
  if (
    [
      "operations",
      "basis",
      "gram",
      "gradient-similarity",
      "vectors",
      "dot",
      "cosine",
      "projection",
    ].includes(concept)
  )
    content = <VectorLab mode={concept as VectorMode} />;
  else if (
    ["matrix", "eigen", "svd", "jacobian", "rank", "basis-change"].includes(
      concept,
    )
  )
    content = <MatrixLab mode={concept as MatrixMode} />;
  else if (
    [
      "descent",
      "manifold",
      "curvature",
      "hessian",
      "field",
      "gradient",
    ].includes(concept)
  )
    content = (
      <SurfaceLab
        mode={concept === "gradient" ? "descent" : (concept as SurfaceMode)}
      />
    );
  else if (concept === "pca") content = <PCALab />;
  else if (
    ["need", "basins", "metrics", "sequential", "inhibition"].includes(concept)
  )
    content = <RoutingLab mode={concept as RoutingMode} />;
  else if (concept === "latent") content = <LatentLab />;
  else if (concept === "shape" || concept === "neighbourhood")
    content = <ShapeLab mode={concept} />;
  else
    content = (
      <RepresentationLab
        mode={concept as "intrinsic" | "neighbours" | "embeddings"}
      />
    );
  return (
    <Suspense fallback={<p className="math-help">Loading geometry lab…</p>}>
      {content}
    </Suspense>
  );
}
