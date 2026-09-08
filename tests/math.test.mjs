import { test } from "node:test";
import assert from "node:assert/strict";
import {
  add,
  cosine,
  descent,
  determinant,
  distance,
  dot,
  gradient,
  loss,
  norm,
  projection,
  subtract,
  trajectory,
  transform,
  matrixMeaning,
} from "../src/research/math/math.ts";
const near = (a, b, tolerance = 1e-9) =>
  assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);
test("vector magnitude, distance and signed dot products", () => {
  assert.equal(norm([3, 4]), 5);
  assert.equal(distance([3, 4], [0, 0]), 5);
  assert.equal(dot([1, 0], [0, 3]), 0);
  assert.equal(dot([2, 0], [-3, 0]), -6);
});
test("cosine is scale invariant and explicitly undefined for zero vectors", () => {
  near(cosine([3, 2], [2, -1]), cosine([6, 4], [6, -3]));
  assert.equal(cosine([0, 0], [1, 0]), null);
  assert.equal(cosine([1, 0], [0, 0]), null);
  assert.equal(cosine([1, 0], [-1, 0]), -1);
});
test("projection residual is orthogonal, including opposed vectors", () => {
  for (const u of [
    [3, 4],
    [-3, 4],
    [0, 0],
  ]) {
    const v = [2, 1],
      p = projection(u, v);
    near(dot(subtract(u, p), v), 0);
    assert.deepEqual(add(p, subtract(u, p)), u);
  }
  assert.equal(projection([2, 1], [0, 0]), null);
});
test("matrix columns, rotation, reflection and collapse use column vectors", () => {
  assert.deepEqual(transform([1, 2, 3, 4], [1, 0]), [1, 3]);
  assert.deepEqual(transform([1, 2, 3, 4], [0, 1]), [2, 4]);
  assert.deepEqual(transform([0, -1, 1, 0], [2, 3]), [-3, 2]);
  assert.equal(determinant([-1, 0, 0, 1]), -1);
  assert.equal(determinant([1, 2, 2, 4]), 0);
  assert.match(matrixMeaning([0, 0, 0, 0]), /single point/);
  assert.match(matrixMeaning([1, 1, 0, 1]), /shear/);
});
test("gradient agrees with finite differences and descent exposes overshoot", () => {
  const p = [2, -3],
    w = [2, 0.5],
    g = gradient(p, w),
    h = 1e-5;
  for (let i = 0; i < 2; i++) {
    const plus = [...p],
      minus = [...p];
    plus[i] += h;
    minus[i] -= h;
    near(g[i], (loss(plus, w) - loss(minus, w)) / (2 * h), 1e-7);
  }
  assert.ok(loss(descent(p, w, 0.1), w) < loss(p, w));
  assert.ok(loss(descent(p, w, 1.2), w) > loss(p, w));
  assert.deepEqual(descent(p, w, 0), p);
});
test("editing an early latent delta propagates to all subsequent states", () => {
  const initial = [-2, -1];
  assert.deepEqual(
    trajectory(initial, [
      [1, 2],
      [2, -0.5],
    ]),
    [
      [-2, -1],
      [-1, 1],
      [1, 0.5],
    ],
  );
  assert.deepEqual(
    trajectory(initial, [
      [2, 2],
      [2, -0.5],
    ]),
    [
      [-2, -1],
      [0, 1],
      [2, 0.5],
    ],
  );
  assert.deepEqual(trajectory(initial, []), [initial]);
});
