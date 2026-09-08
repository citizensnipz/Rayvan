import { test } from "node:test";
import assert from "node:assert/strict";
import {
  covariance,
  length,
  minus,
  cosine3,
} from "../src/research/math/core/geometry.ts";
import {
  nearestSamples,
  dimensionSamples,
  varianceSpectrum,
  normalizeEmbedding,
  pcaProjector,
  embeddingSamples,
} from "../src/research/math/core/representations.ts";
const close = (a, b, eps = 1e-9) =>
  assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

test("nearest samples retain source IDs, exclude self and handle undefined cosine", () => {
  const points = [
    [0, 0, 0],
    [10, 0, 0],
    [1, 1, 0],
    [1, 0, 0],
  ];
  assert.equal(
    nearestSamples(points, points[3], 1, "euclidean", 3)[0].index,
    0,
  );
  assert.equal(nearestSamples(points, points[3], 1, "cosine", 3)[0].index, 1);
  assert.deepEqual(nearestSamples(points, [0, 0, 0], 4, "cosine"), []);
  assert.equal(nearestSamples(points, [1, 0, 0], 9, "cosine").length, 3);
  assert.deepEqual(nearestSamples(points, [1, 0, 0], 0), []);
});
test("line and sheet have exact covariance ranks, curved sheet has full global rank", () => {
  for (const [kind, rank] of [
    ["line", 1],
    ["sheet", 2],
    ["volume", 3],
    ["curved", 3],
  ]) {
    const stats = covariance(dimensionSamples(kind, 0));
    assert.equal(stats.values.filter((v) => v > 1e-9).length, rank, kind);
  }
  close(
    varianceSpectrum(covariance(dimensionSamples("line", 0)).values)
      .participation,
    1,
  );
  close(
    varianceSpectrum(covariance(dimensionSamples("sheet", 0)).values)
      .participation,
    2,
  );
  assert.equal(
    covariance(dimensionSamples("sheet", 0.2)).values.filter((v) => v > 1e-9)
      .length,
    3,
  );
  assert.equal(
    covariance(dimensionSamples("volume", 0, 0)).values.filter((v) => v > 1e-9)
      .length,
    1,
  );
});
test("variance spectrum is scale invariant and safe for collapsed clouds", () => {
  close(varianceSpectrum([1, 1, 1]).participation, 3);
  close(
    varianceSpectrum([20, 10, 0]).participation,
    varianceSpectrum([2, 1, 0]).participation,
  );
  assert.equal(varianceSpectrum([99, 1, 0]).dimensions95, 1);
  assert.deepEqual(varianceSpectrum([0, 0, 0]).shares, [0, 0, 0]);
  assert.equal(varianceSpectrum([0, 0, 0]).participation, 0);
});
test("normalization preserves cosine and connects angular to Euclidean distance", () => {
  const a = [2, 1, 3],
    b = [-1, 2, 0.5];
  for (const t of [0, 0.2, 0.7, 1])
    close(
      cosine3(normalizeEmbedding(a, t), normalizeEmbedding(b, t)),
      cosine3(a, b),
    );
  const u = normalizeEmbedding(a, 1),
    v = normalizeEmbedding(b, 1);
  close(length(u), 1);
  close(length(minus(u, v)) ** 2, 2 * (1 - cosine3(a, b)));
  assert.deepEqual(normalizeEmbedding([0, 0, 0], 1), [0, 0, 0]);
});
test("PCA projection centres query consistently and reports discarded variance", () => {
  const samples = embeddingSamples(),
    points = samples.map((s) => s.coordinates);
  assert.equal(new Set(samples.map((s) => s.id)).size, points.length);
  const full = pcaProjector(points, 3),
    reduced = pcaProjector(points, 2);
  close(
    length(minus(full.project(points[0]), full.project(points[60]))),
    length(minus(points[0], points[60])),
  );
  assert.equal(reduced.project(points[0])[2], 0);
  close(length(full.project(full.stats.mean)), 0);
  const mse =
    points.reduce((sum, p) => sum + full.project(p)[2] ** 2, 0) /
    (points.length - 1);
  close(mse, full.stats.values[2]);
});
