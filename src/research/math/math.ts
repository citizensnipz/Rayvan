import type { Matrix2, Vec2 } from "./types";
export const add = (u: Vec2, v: Vec2): Vec2 => [u[0] + v[0], u[1] + v[1]];
export const subtract = (u: Vec2, v: Vec2): Vec2 => [u[0] - v[0], u[1] - v[1]];
export const scale = (v: Vec2, n: number): Vec2 => [v[0] * n, v[1] * n];
export const dot = (u: Vec2, v: Vec2) => u[0] * v[0] + u[1] * v[1];
export const norm = (v: Vec2) => Math.hypot(...v);
export const distance = (u: Vec2, v: Vec2) => norm(subtract(u, v));
export const cosine = (u: Vec2, v: Vec2): number | null =>
  norm(u) === 0 || norm(v) === 0
    ? null
    : Math.max(-1, Math.min(1, dot(u, v) / norm(u) / norm(v)));
export const projection = (u: Vec2, onto: Vec2): Vec2 | null =>
  norm(onto) === 0 ? null : scale(onto, dot(u, onto) / dot(onto, onto));
export const transform = ([a, b, c, d]: Matrix2, [x, y]: Vec2): Vec2 => [
  a * x + b * y,
  c * x + d * y,
];
export const determinant = ([a, b, c, d]: Matrix2) => a * d - b * c;
export const loss = ([x, y]: Vec2, [a, b]: Vec2) => a * x * x + b * y * y;
export const gradient = ([x, y]: Vec2, [a, b]: Vec2): Vec2 => [
  2 * a * x,
  2 * b * y,
];
export const descent = (p: Vec2, weights: Vec2, rate: number) =>
  subtract(p, scale(gradient(p, weights), rate));
export const trajectory = (initial: Vec2, deltas: readonly Vec2[]): Vec2[] =>
  deltas.reduce<Vec2[]>(
    (points, delta) => [...points, add(points[points.length - 1], delta)],
    [initial],
  );
export const fmt = (n: number) =>
  Math.abs(n) < 0.00005 ? "0" : Number(n.toFixed(4)).toString();
export const coords = (v: Vec2) => `(${fmt(v[0])}, ${fmt(v[1])})`;
export function matrixMeaning(m: Matrix2) {
  const det = determinant(m),
    first = transform(m, [1, 0]),
    second = transform(m, [0, 1]);
  if (Math.abs(det) < 1e-9)
    return `This matrix collapses the plane onto ${norm(first) + norm(second) < 1e-9 ? "a single point" : "a line"}. Information is lost: the transformation cannot be inverted.`;
  const orthogonal = Math.abs(dot(first, second)) < 1e-9;
  const unit =
    Math.abs(norm(first) - 1) < 1e-9 && Math.abs(norm(second) - 1) < 1e-9;
  const action = orthogonal
    ? unit
      ? det < 0
        ? "reflects the plane"
        : Math.abs(m[0] - 1) < 1e-9 && Math.abs(m[3] - 1) < 1e-9
          ? "leaves the plane unchanged"
          : "rotates the plane"
      : "scales along perpendicular transformed axes, possibly with rotation or reflection"
    : "makes the basis oblique (shear), possibly combined with rotation and scaling";
  return `This matrix ${action}. Areas are multiplied by ${fmt(Math.abs(det))}; orientation is ${det < 0 ? "reversed" : "preserved"}.`;
}
