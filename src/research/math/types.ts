export type Vec2 = readonly [number, number];
export type Matrix2 = readonly [number, number, number, number];
export type ConceptId =
  | "vectors"
  | "norm"
  | "dot"
  | "cosine"
  | "distance"
  | "projection"
  | "matrix"
  | "eigen"
  | "pca"
  | "svd"
  | "gradient"
  | "jacobian"
  | "hessian"
  | "latent"
  | "operations"
  | "basis"
  | "gram"
  | "basis-change"
  | "rank"
  | "descent"
  | "field"
  | "manifold"
  | "curvature"
  | "embeddings"
  | "intrinsic"
  | "neighbours"
  | "need"
  | "basins"
  | "metrics"
  | "sequential"
  | "inhibition"
  | "shape"
  | "gradient-similarity"
  | "neighbourhood"
  | "umap";
export interface Frame<T> {
  step: number;
  state: T;
  label?: string;
}
/** A future adapter can supply projected states without coupling rendering to run files. */
export interface Trajectory<T> {
  frames: readonly Frame<T>[];
  source: "manual" | "experiment";
  runId?: string;
  projection?: {
    method: string;
    originalDimensions: number;
    axisLabels: readonly [string, string];
  };
}
export interface VectorState {
  u: Vec2;
  v: Vec2;
}
export interface GradientState {
  point: Vec2;
  weights: Vec2;
  rate: number;
}
export interface LatentState {
  initial: Vec2;
  deltas: readonly Vec2[];
}
