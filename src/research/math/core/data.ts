import type { V3, M3 } from "./geometry";
export interface ProjectedState {
  id: string;
  coordinates: V3;
  original?: readonly number[];
  step?: number;
  tokenId?: string;
  selectedExpert?: string;
  routingScores?: Record<string, number>;
  loss?: number;
  regret?: number;
}
export interface RepresentationData {
  source: "toy" | "projected";
  originalDimensions: number;
  projection: {
    method: "raw" | "pca" | "custom";
    axes: readonly number[];
    explainedVariance?: number[];
  };
  states: readonly ProjectedState[];
  prototypes?: readonly {
    expertId: string;
    center: V3;
    width?: number;
    covariance?: M3;
  }[];
}
export function validateRepresentation(data: RepresentationData) {
  if (!Number.isInteger(data.originalDimensions) || data.originalDimensions < 2)
    throw new Error("Invalid original dimensionality");
  for (const state of data.states)
    if (
      state.coordinates.length !== 3 ||
      !state.coordinates.every(Number.isFinite)
    )
      throw new Error("Nonfinite projected coordinates");
  return data;
}
