import { useState } from "react";
import type { ConceptId } from "./types";
import { MathCategorySidebar } from "./components/MathCategorySidebar";
import { VectorVisualization } from "./visualizations/VectorVisualizations";
import { MatrixTransformation } from "./visualizations/MatrixTransformation";
import { GradientVisualization } from "./visualizations/GradientVisualization";
import { LatentTransformation } from "./visualizations/LatentTransformation";
import "./math.css";
export function MathVisualizerPage() {
  const [concept, setConcept] = useState<ConceptId>("vectors");
  return (
    <div className="math-lab">
      <MathCategorySidebar selected={concept} onSelect={setConcept} />
      <div className="math-module">
        {concept === "matrix" ? (
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
