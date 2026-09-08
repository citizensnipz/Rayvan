export interface ProjectionSelection {
  method: "raw" | "pca" | "custom";
  axes: readonly [number, number, number];
}
export function ProjectionControls({
  value,
  onChange,
  dimensions = 3,
  hasCustom = false,
}: {
  value: ProjectionSelection;
  onChange: (value: ProjectionSelection) => void;
  dimensions?: number;
  hasCustom?: boolean;
}) {
  return (
    <fieldset className="math-vector-controls">
      <legend>Display coordinates</legend>
      <label>
        <span>Projection method</span>
        <select
          value={value.method}
          onChange={(e) =>
            onChange({
              ...value,
              method: e.target.value as ProjectionSelection["method"],
            })
          }
        >
          <option value="raw">Raw dimensions</option>
          <option value="pca">PCA components</option>
          <option value="custom" disabled={!hasCustom}>
            Custom coordinates{hasCustom ? "" : " (supply data)"}
          </option>
          <option disabled>UMAP — Coming Soon</option>
        </select>
      </label>
      {value.method === "raw" &&
        value.axes.map((axis, i) => (
          <label key={i}>
            <span>Display {["X", "Y", "Z"][i]}</span>
            <select
              value={axis}
              onChange={(e) =>
                onChange({
                  ...value,
                  axes: value.axes.map((a, j) =>
                    i === j ? Number(e.target.value) : a,
                  ) as unknown as ProjectionSelection["axes"],
                })
              }
            >
              {Array.from({ length: dimensions }, (_, d) => (
                <option key={d} value={d}>
                  Dimension {d + 1}
                </option>
              ))}
            </select>
          </label>
        ))}
    </fieldset>
  );
}
