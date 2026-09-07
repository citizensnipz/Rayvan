import {
  cosine3,
  inner,
  length,
  minus,
  mv,
  plus,
  times,
  type V3,
  type M3,
} from "./geometry";
export interface ToyExpert {
  id: string;
  center: V3;
  width: number;
  matrix: M3;
  shift: V3;
  color: string;
}
export type Metric = "euclidean" | "cosine" | "mahalanobis";
export const experts: ToyExpert[] = [
  {
    id: "Rotate",
    center: [-1, 1, 0],
    width: 1,
    matrix: [
      [0.85, -0.5, 0],
      [0.5, 0.85, 0],
      [0, 0, 1],
    ],
    shift: [0.6, 0, 0.1],
    color: "#38c6cc",
  },
  {
    id: "Contract",
    center: [1, 1, 0.4],
    width: 1,
    matrix: [
      [0.65, 0, 0],
      [0, 0.65, 0],
      [0, 0, 0.65],
    ],
    shift: [0, -0.8, 0],
    color: "#b299ff",
  },
  {
    id: "Shear",
    center: [0, -1, -0.4],
    width: 1,
    matrix: [
      [1, 0.4, 0],
      [0, 1, 0],
      [0, 0, 0.8],
    ],
    shift: [-0.7, 0.4, 0],
    color: "#d8ff75",
  },
];
export function metricDistance(
  a: V3,
  b: V3,
  metric: Metric,
  variance: V3 = [0.3, 2, 1],
): number | null {
  if (metric === "cosine") {
    const c = cosine3(a, b);
    return c === null ? null : 1 - c;
  }
  const d = minus(a, b);
  return metric === "mahalanobis"
    ? Math.sqrt(d.reduce((s, n, i) => s + (n * n) / variance[i], 0))
    : length(d);
}
export function rankExperts(
  z: V3,
  list: readonly ToyExpert[],
  metric: Metric,
  inhibition: Record<string, number> = {},
) {
  return list
    .map((expert) => {
      const distance = metricDistance(z, expert.center, metric);
      const base =
        distance === null
          ? -Infinity
          : -(distance ** 2) / (2 * expert.width ** 2);
      return {
        expert,
        distance,
        base,
        penalty: inhibition[expert.id] ?? 0,
        score: base - (inhibition[expert.id] ?? 0),
      };
    })
    .sort((a, b) => b.score - a.score);
}
export function routeSequence(
  initial: V3,
  list: readonly ToyExpert[],
  metric: Metric,
  steps: number,
  strength: number,
  decay: number,
) {
  let z = initial;
  const last: Record<string, number> = {};
  const frames = [];
  for (let step = 0; step < steps; step++) {
    const inhibition = Object.fromEntries(
      Object.entries(last).map(([id, t]) => [
        id,
        strength * Math.exp(-decay * (step - t - 1)),
      ]),
    );
    const ranks = rankExperts(z, list, metric, inhibition),
      selected = ranks[0];
    if (!selected || !Number.isFinite(selected.score)) break;
    const next = plus(mv(selected.expert.matrix, z), selected.expert.shift),
      delta = minus(next, z);
    frames.push({ z, next, delta, selected: selected.expert.id, ranks, step });
    last[selected.expert.id] = step;
    z = next;
    if (length(z) > 30) break;
  }
  return frames;
}
export const shapeError = (
  desired: M3,
  actual: M3,
  points: readonly V3[],
  shift: V3 = [0, 0, 0],
  desiredShift: V3 = [0, 0, 0],
) =>
  points.reduce((s, p) => {
    const d = minus(
      plus(mv(desired, p), desiredShift),
      plus(mv(actual, p), shift),
    );
    return s + inner(d, d);
  }, 0) / Math.max(1, points.length);
