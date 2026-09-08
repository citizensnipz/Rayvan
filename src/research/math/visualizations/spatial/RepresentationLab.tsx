import { EmbeddingCloudLab } from "../representations/EmbeddingCloudLab";
import { IntrinsicDimensionLab } from "../representations/IntrinsicDimensionLab";
import { NeighbourhoodLab } from "../representations/NeighbourhoodLab";

/** Shared category entry point; independent experiments and state. */
export function RepresentationLab({
  mode = "intrinsic",
}: {
  mode?: "intrinsic" | "neighbours" | "embeddings";
}) {
  if (mode === "embeddings") return <EmbeddingCloudLab />;
  if (mode === "neighbours") return <NeighbourhoodLab />;
  return <IntrinsicDimensionLab />;
}
