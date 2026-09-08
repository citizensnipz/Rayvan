import { useState } from "react";
import type { ConceptId } from "./types";
import { MathCategorySidebar } from "./components/MathCategorySidebar";
import { VectorVisualization } from "./visualizations/VectorVisualizations";
import { MatrixTransformation } from "./visualizations/MatrixTransformation";
import { GradientVisualization } from "./visualizations/GradientVisualization";
import { LatentTransformation } from "./visualizations/LatentTransformation";
import "./math.css";
import { SpatialVisualizer } from "./SpatialVisualizer";
export function MathVisualizerPage() {
  const [concept, setConcept] = useState<ConceptId>("vectors");
  const [spatial, setSpatial] = useState(false);
  const legacy = [
    "vectors",
    "norm",
    "dot",
    "cosine",
    "distance",
    "projection",
    "matrix",
    "gradient",
    "latent",
  ].includes(concept);
  const supportsSpatial = legacy && !["norm", "distance"].includes(concept);
  return (
    <div className="math-lab">
      <MathCategorySidebar selected={concept} onSelect={setConcept} />
      <div className="math-module">
        {supportsSpatial && (
          <div
            className="math-view-switch"
            role="group"
            aria-label="Workspace renderer"
          >
            <button aria-pressed={!spatial} onClick={() => setSpatial(false)}>
              Original 2D
            </button>
            <button aria-pressed={spatial} onClick={() => setSpatial(true)}>
              Spatial 2D / 3D
            </button>
          </div>
        )}
        {!legacy || (supportsSpatial && spatial) ? (
          <SpatialVisualizer key={concept} concept={concept} />
        ) : concept === "matrix" ? (
          <MatrixTransformation />
        ) : concept === "gradient" ? (
          <GradientVisualization />
        ) : concept === "latent" ? (
          <LatentTransformation />
        ) : [
            "vectors",
            "norm",
            "dot",
            "cosine",
            "distance",
            "projection",
          ].includes(concept) ? (
          <VectorVisualization
            key={concept}
            mode={
              concept as
                | "vectors"
                | "norm"
                | "dot"
                | "cosine"
                | "distance"
                | "projection"
            }
          />
        ) : null}
      </div>
    </div>
  );
}
