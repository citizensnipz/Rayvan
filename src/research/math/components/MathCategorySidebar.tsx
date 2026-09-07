import type { ConceptId } from "../types";
export const categories: {
  title: string;
  items: { id: ConceptId; label: string; later?: boolean }[];
}[] = [
  {
    title: "Foundational",
    items: [
      { id: "vectors", label: "Vectors" },
      { id: "norm", label: "Vector magnitude / norm" },
      { id: "dot", label: "Dot product" },
      { id: "cosine", label: "Cosine similarity" },
      { id: "distance", label: "Euclidean distance" },
      { id: "projection", label: "Vector projection" },
      { id: "matrix", label: "Matrix transformation" },
    ],
  },
  {
    title: "Advanced",
    items: [
      { id: "eigen", label: "Eigenvectors / eigenvalues", later: true },
      { id: "pca", label: "PCA", later: true },
      { id: "svd", label: "SVD", later: true },
      { id: "gradient", label: "Gradient" },
      { id: "jacobian", label: "Jacobian", later: true },
      { id: "hessian", label: "Hessian / curvature", later: true },
    ],
  },
  { title: "EMC", items: [{ id: "latent", label: "Latent Transformation" }] },
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
              {item.later && <small>Coming later</small>}
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
