import type { ConceptId } from "../types";
export const categories: {
  title: string;
  items: { id: ConceptId; label: string; later?: boolean }[];
}[] = [
  {
    title: "Foundations",
    items: [
      { id: "vectors", label: "Vectors" },
      { id: "norm", label: "Vector magnitude / norm" },
      { id: "operations", label: "Addition / subtraction / scaling" },
      { id: "basis", label: "Basis / linear combinations" },
      { id: "gram", label: "Orthogonality / Gram–Schmidt" },
      { id: "dot", label: "Dot product" },
      { id: "cosine", label: "Cosine similarity" },
      { id: "distance", label: "Euclidean distance" },
      { id: "projection", label: "Vector projection" },
      { id: "matrix", label: "Matrix transformation" },
    ],
  },
  {
    title: "Linear Algebra / Geometry",
    items: [
      { id: "basis-change", label: "Change of basis" },
      { id: "eigen", label: "Eigenvectors / eigenvalues" },
      { id: "pca", label: "PCA / covariance" },
      { id: "svd", label: "SVD / singular spectrum" },
      { id: "rank", label: "Rank / null space / volume" },
    ],
  },
  {
    title: "Optimization / Geometry",
    items: [
      { id: "gradient", label: "Gradient" },
      { id: "descent", label: "3D gradient descent" },
      { id: "jacobian", label: "Jacobian geometry" },
      { id: "hessian", label: "Hessian" },
      { id: "field", label: "Vector fields" },
      { id: "manifold", label: "Manifold / tangent / paths" },
      { id: "curvature", label: "Curvature" },
    ],
  },
  {
    title: "Representation Spaces",
    items: [
      { id: "embeddings", label: "Embedding clouds / projection" },
      { id: "neighbours", label: "Local neighbourhoods" },
      { id: "intrinsic", label: "Intrinsic dimension" },
      { id: "umap", label: "UMAP", later: true },
    ],
  },
  {
    title: "EMC Geometry",
    items: [
      { id: "latent", label: "Latent Transformation" },
      { id: "need", label: "Computational need" },
      { id: "basins", label: "Competence basins" },
      { id: "metrics", label: "Routing distance metrics" },
      { id: "sequential", label: "Sequential routing" },
      { id: "inhibition", label: "Refractory / inhibition" },
      { id: "shape", label: "Transformation shape" },
      { id: "gradient-similarity", label: "Gradient-direction similarity" },
      { id: "neighbourhood", label: "Neighbourhood evolution" },
    ],
  },
];
export function MathCategorySidebar({
  selected,
  onSelect,
}: {
  selected: ConceptId;
  onSelect: (id: ConceptId) => void;
}) {
  return (
    <nav className="math-categories" aria-label="Math concepts">
      {categories.map((category) => (
        <section key={category.title}>
          <h2>{category.title}</h2>
          {category.items.map((item) => (
            <button
              key={item.id}
              disabled={item.later}
              aria-current={selected === item.id ? "page" : undefined}
              onClick={() => onSelect(item.id)}
            >
              <span>{item.label}</span>
              {item.later && <small>Coming Soon</small>}
            </button>
          ))}
        </section>
      ))}
      <p className="math-help">
        Manual values · immediate feedback
        <br />
        No training run required
      </p>
    </nav>
  );
}
