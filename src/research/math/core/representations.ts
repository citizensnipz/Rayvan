import {
  cosine3,
  covariance,
  length,
  minus,
  mv,
  plus,
  times,
  unit,
  type V3,
} from "./geometry";

export type NeighbourMetric = "euclidean" | "cosine";
export interface EmbeddingSample {
  id: string;
  group: number;
  coordinates: V3;
}

/** Deterministic toy clusters, not real tokens or a learned embedding. */
export function embeddingSamples(
  separation = 1,
  spread = 0.3,
): EmbeddingSample[] {
  const centres: V3[] = [
    [0.8, 0.2, 0.25],
    [2.6, 0.65, 0.8],
    [-0.8, 1.6, -0.6],
  ];
  return centres.flatMap((centre, group) =>
    Array.from({ length: 40 }, (_, i) => ({
      id: `c${group + 1}-${i}`,
      group,
      coordinates: plus(
        times(centre, separation),
        times(
          [
            Math.sin(i * 2.399 + group),
            Math.cos(i * 1.73 + group),
            Math.sin(i * 0.91 + group),
          ],
          spread,
        ),
      ),
    })),
  );
}

/** Stable source indices survive sorting and exclusion. Undefined cosine is excluded. */
export function nearestSamples(
  points: readonly V3[],
  query: V3,
  k: number,
  metric: NeighbourMetric = "euclidean",
  excludeIndex = -1,
) {
  return points
    .flatMap((point, index) => {
      if (index === excludeIndex) return [];
      const similarity = metric === "cosine" ? cosine3(point, query) : null;
      const distance =
        metric === "euclidean"
          ? length(minus(point, query))
          : similarity === null
            ? null
            : 1 - similarity;
      return distance === null
        ? []
        : [{ index, point, distance: Math.max(0, distance) }];
    })
    .sort((a, b) => a.distance - b.distance || a.index - b.index)
    .slice(0, Math.max(0, Math.floor(k)));
}

/** Global LINEAR spread proxy, not an intrinsic-dimension estimator. */
export function varianceSpectrum(values: readonly number[]) {
  const positive = values.map((v) => Math.max(0, v));
  const total = positive.reduce((a, b) => a + b, 0);
  const shares = positive.map((v) => (total > 1e-12 ? v / total : 0));
  const squared = shares.reduce((a, b) => a + b * b, 0);
  let cumulative = 0,
    dimensions95 = 0;
  if (total > 1e-12)
    for (const share of shares) {
      cumulative += share;
      dimensions95++;
      if (cumulative >= 0.95 - 1e-12) break;
    }
  return {
    shares,
    total,
    participation: squared > 0 ? 1 / squared : 0,
    dimensions95,
  };
}

export type IntrinsicShape = "line" | "sheet" | "curved" | "volume";
export function dimensionSamples(
  kind: IntrinsicShape,
  thickness: number,
  unfold = 1,
): V3[] {
  const e1: V3 = [Math.SQRT1_2, Math.SQRT1_2, 0],
    e2: V3 = [-0.5, 0.5, Math.SQRT1_2],
    e3: V3 = [0.5, -0.5, Math.SQRT1_2];
  const result: V3[] = [];
  for (let i = 0; i < 7; i++)
    for (let j = 0; j < 7; j++)
      for (let k = 0; k < 3; k++) {
        const u = (i - 3) * 0.55,
          v = (j - 3) * 0.55,
          w = (k - 1) * 1.65;
        const b = kind === "line" ? thickness * v : unfold * v;
        const c =
          kind === "volume"
            ? unfold * w
            : thickness * w +
              (kind === "curved" ? unfold * 0.65 * (u * u + v * v) : 0);
        result.push(plus(plus(times(e1, u), times(e2, b)), times(e3, c)));
      }
  return result;
}

export function normalizeEmbedding(point: V3, amount: number): V3 {
  const normalized = unit(point);
  return normalized
    ? plus(times(point, 1 - amount), times(normalized, amount))
    : point;
}

export function pcaProjector(points: readonly V3[], retain: 2 | 3) {
  const stats = covariance(points);
  return {
    stats,
    project: (point: V3): V3 => {
      const p = mv(stats.vectors, minus(point, stats.mean));
      return retain === 2 ? [p[0], p[1], 0] : p;
    },
  };
}
