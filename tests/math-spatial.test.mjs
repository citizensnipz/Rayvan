import { test } from "node:test";
import assert from "node:assert/strict";
import {
  I3,
  cosine3,
  covariance,
  det3,
  eigen2,
  inner,
  length,
  minus,
  mm,
  mv,
  svd3,
  symmetricEigen,
  transpose,
  unit,
  cloud,
  orthogonalStep,
} from "../src/research/math/core/geometry.ts";
import {
  curvatures,
  derivatives,
  manifoldPath,
  optimization,
  surface,
} from "../src/research/math/core/differential.ts";
import {
  experts,
  metricDistance,
  rankExperts,
  routeSequence,
  shapeError,
} from "../src/research/math/core/routing.ts";
import { validateRepresentation } from "../src/research/math/core/data.ts";
const near = (a, b, e = 1e-7) => assert.ok(Math.abs(a - b) < e, `${a} ≠ ${b}`);
const matrixNear = (a, b, e = 1e-7) =>
  a.flat().forEach((x, i) => near(x, b.flat()[i], e));
test("symmetric eigensolver has orthonormal eigenvectors and correct eigenpairs", () => {
  const a = [
      [2, 1, 0.3],
      [1, 4, -0.7],
      [0.3, -0.7, 1],
    ],
    result = symmetricEigen(a);
  matrixNear(mm(result.vectors, transpose(result.vectors)), I3);
  result.vectors.forEach((v, i) =>
    mv(a, v).forEach((x, j) => near(x, result.values[i] * v[j])),
  );
});
test("SVD reconstructs dense, reflected, singular, repeated and zero matrices", () => {
  const examples = [
    I3,
    [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ],
    [
      [1, 2, 3],
      [2, 4, 6],
      [0, 0, 0],
    ],
    [
      [-1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ],
    [
      [0, -1, 0],
      [1, 0, 0],
      [0, 0, 1],
    ],
  ];
  for (let seed = 0; seed < 40; seed++)
    examples.push(
      I3.map((_, i) =>
        I3.map((_, j) => Math.sin(seed * 11 + i * 7 + j * 3) * 2),
      ),
    );
  for (const a of examples) {
    const { u, s, vt } = svd3(a),
      sigma = [
        [s[0], 0, 0],
        [0, s[1], 0],
        [0, 0, s[2]],
      ];
    matrixNear(mm(mm(u, sigma), vt), a, 1e-6);
    matrixNear(mm(transpose(u), u), I3);
    matrixNear(mm(vt, transpose(vt)), I3);
    assert.ok(s[0] >= s[1] && s[1] >= s[2]);
  }
});
test("PCA centres samples, diagonalizes covariance and explains expected intrinsic ranks", () => {
  for (const [kind, rank] of [
    ["line", 1],
    ["sheet", 2],
    ["volume", 3],
  ]) {
    const p = cloud(kind, 0),
      c = covariance(p);
    assert.equal(c.values.filter((v) => v > 1e-8).length, rank);
    for (let j = 0; j < 3; j++)
      near(
        c.centered.reduce((s, v) => s + v[j], 0),
        0,
      );
    const rotated = covariance(c.centered.map((p) => mv(c.vectors, p)));
    near(rotated.matrix[0][1], 0);
    near(rotated.matrix[0][2], 0);
  }
});
test("zero directions are undefined; real eigenvectors satisfy Av=λv", () => {
  assert.equal(unit([0, 0, 0]), null);
  assert.equal(cosine3([0, 0, 0], [1, 0, 0]), null);
  assert.equal(eigen2(0, -1, 1, 0).complex, true);
  for (const a of [
    [
      [2, 1, 0],
      [1, 1, 0],
      [0, 0, 1],
    ],
    [
      [1, 1, 0],
      [0, 1, 0],
      [0, 0, 1],
    ],
    I3,
  ])
    for (const p of eigen2(a[0][0], a[0][1], a[1][0], a[1][1]).pairs)
      mv(a, p.vector).forEach((x, i) => near(x, p.value * p.vector[i]));
});
test("analytic gradients and Hessians match finite differences for every landscape", () => {
  const x = 0.7,
    y = -0.4,
    h = 1e-4;
  for (const kind of [
    "bowl",
    "valley",
    "saddle",
    "multi-basin",
    "plane",
    "wave",
  ]) {
    const d = derivatives(kind, x, y);
    near(
      d.gx,
      (surface(kind, x + h, y) - surface(kind, x - h, y)) / (2 * h),
      1e-6,
    );
    near(
      d.gy,
      (surface(kind, x, y + h) - surface(kind, x, y - h)) / (2 * h),
      1e-6,
    );
    near(
      d.xx,
      (derivatives(kind, x + h, y).gx - derivatives(kind, x - h, y).gx) /
        (2 * h),
      1e-6,
    );
    near(
      d.xy,
      (derivatives(kind, x, y + h).gx - derivatives(kind, x, y - h).gx) /
        (2 * h),
      1e-6,
    );
    near(
      d.yy,
      (derivatives(kind, x, y + h).gy - derivatives(kind, x, y - h).gy) /
        (2 * h),
      1e-6,
    );
  }
});
test("surface curvature distinguishes flat, elliptic and saddle geometry", () => {
  near(curvatures("plane", 0, 0).gaussian, 0);
  assert.ok(curvatures("bowl", 0, 0).gaussian > 0);
  assert.ok(curvatures("saddle", 0, 0).gaussian < 0);
  const path = manifoldPath("bowl", [-2, -1, 0], [2, 1, 0]);
  assert.ok(path.distance >= length(minus(path.points[0], path.points.at(-1))));
});
test("optimization recomputes loss and explicitly stops before leaving domain", () => {
  const result = optimization("bowl", [2, 1, 0], 0.2, 10);
  assert.equal(result.points.length, 11);
  result.points
    .slice(1)
    .forEach((p, i) => assert.ok(p[2] < result.points[i][2]));
  assert.equal(optimization("saddle", [0, 3, 0], 1.2, 20).stopped, true);
});
test("metric choice can change the winner and zero cosine has no finite winner", () => {
  const a = { ...experts[0], id: "a", center: [1, 0, 0] },
    b = { ...experts[1], id: "b", center: [9, 2, 0] };
  assert.equal(rankExperts([10, 0, 0], [a, b], "euclidean")[0].expert.id, "b");
  assert.equal(rankExperts([10, 0, 0], [a, b], "cosine")[0].expert.id, "a");
  assert.equal(metricDistance([0, 0, 0], [1, 0, 0], "cosine"), null);
  assert.equal(routeSequence([0, 0, 0], [a, b], "cosine", 4, 0, 1).length, 0);
  near(metricDistance([1, 0, 0], [0, 0, 0], "mahalanobis"), Math.sqrt(1 / 0.3));
});
test("sequential routing recomputes state and inhibition decays rather than excludes", () => {
  const list = experts.map((e, i) => ({
    ...e,
    center: [0, 0, 0],
    matrix: I3,
    shift: [0, 0, 0],
  }));
  const frames = routeSequence([1, 0, 0], list, "euclidean", 5, 0.6, 1);
  assert.notEqual(frames[1].selected, frames[0].selected);
  near(
    frames[1].ranks.find((r) => r.expert.id === frames[0].selected).penalty,
    0.6,
  );
  near(
    frames[2].ranks.find((r) => r.expert.id === frames[0].selected).penalty,
    0.6 * Math.exp(-1),
  );
  frames.forEach((f) =>
    assert.ok(f.ranks.every((r) => Number.isFinite(r.score))),
  );
  const moving = routeSequence([-2, 0.5, 0], experts, "euclidean", 4, 0, 1);
  moving.slice(1).forEach((f, i) => assert.deepEqual(f.z, moving[i].next));
});
test("shape matching is exact for the target and projection data rejects NaN", () => {
  near(
    shapeError(I3, I3, [
      [1, 0, 0],
      [0, 1, 0],
    ]),
    0,
  );
  assert.ok(shapeError(I3, I3, [[1, 0, 0]], [1, 0, 0]) > 0);
  assert.throws(() =>
    validateRepresentation({
      source: "projected",
      originalDimensions: 8,
      projection: { method: "custom", axes: [0, 1, 2] },
      states: [{ id: "a", coordinates: [NaN, 0, 0] }],
    }),
  );
});
test("orthogonal animation preserves lengths for rotations and reaches reflected endpoints", () => {
  for (const q of [
    [
      [0, -1, 0],
      [1, 0, 0],
      [0, 0, 1],
    ],
    [
      [-1, 0, 0],
      [0, -1, 0],
      [0, 0, 1],
    ],
  ]) {
    matrixNear(orthogonalStep(q, 0), I3);
    matrixNear(orthogonalStep(q, 1), q);
    for (const t of [0.2, 0.5, 0.8])
      matrixNear(mm(transpose(orthogonalStep(q, t)), orthogonalStep(q, t)), I3);
  }
  const reflected = [
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  matrixNear(orthogonalStep(reflected, 1), reflected);
});
