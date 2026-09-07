export type V3 = readonly [number, number, number];
/** Row-major; vectors are columns. */
export type M3 = readonly [V3, V3, V3];
export const I3: M3 = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];
export const plus = (a: V3, b: V3): V3 => [
  a[0] + b[0],
  a[1] + b[1],
  a[2] + b[2],
];
export const times = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
export const minus = (a: V3, b: V3) => plus(a, times(b, -1));
export const inner = (a: V3, b: V3) => a.reduce((s, x, i) => s + x * b[i], 0);
export const length = (a: V3) => Math.hypot(...a);
export const unit = (a: V3): V3 | null =>
  length(a) < 1e-12 ? null : times(a, 1 / length(a));
export const cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const cosine3 = (a: V3, b: V3): number | null =>
  unit(a) && unit(b)
    ? Math.max(-1, Math.min(1, inner(unit(a)!, unit(b)!)))
    : null;
export const mv = (a: M3, v: V3): V3 =>
  a.map((r) => inner(r, v)) as unknown as V3;
export const transpose = (a: M3): M3 =>
  I3.map((_, i) => a.map((r) => r[i])) as unknown as M3;
export const mm = (a: M3, b: M3): M3 =>
  a.map((r) => transpose(b).map((c) => inner(r, c))) as unknown as M3;
export const mix = (a: V3, b: V3, t: number) =>
  plus(times(a, 1 - t), times(b, t));
export const mixMatrix = (a: M3, b: M3, t: number): M3 =>
  a.map((r, i) => mix(r, b[i], t)) as unknown as M3;
export const det3 = (a: M3) => inner(a[0], cross(a[1], a[2]));
/** Rotate continuously through SO(3); if needed, also interpolate an explicit z reflection. */
export function orthogonalStep(q: M3, t: number): M3 {
  const reflected = det3(q) < 0,
    reflection: M3 = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, reflected ? -1 : 1],
    ];
  const r = mm(q, reflection),
    angle = Math.acos(
      Math.max(-1, Math.min(1, (r[0][0] + r[1][1] + r[2][2] - 1) / 2)),
    );
  let axis: V3 = [1, 0, 0];
  if (angle > 1e-8) {
    if (Math.PI - angle < 1e-6) {
      const candidates = transpose(
        r.map((row, i) =>
          row.map((x, j) => x + (i === j ? 1 : 0)),
        ) as unknown as M3,
      );
      axis = unit([...candidates].sort((a, b) => length(b) - length(a))[0])!;
    } else
      axis = unit([r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]])!;
  }
  const c = Math.cos(angle * t),
    s = Math.sin(angle * t),
    [x, y, z] = axis;
  const rotation: M3 = [
    [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
    [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
    [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
  ];
  return mm(rotation, [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, reflected ? 1 - 2 * t : 1],
  ]);
}
export const f3 = (a: V3) =>
  `(${a.map((n) => Number(n.toFixed(3))).join(", ")})`;
export const format = (n: number | null) =>
  n === null || !Number.isFinite(n)
    ? "Undefined"
    : Number(n.toFixed(4)).toString();

/** Jacobi eigensolver for real symmetric 3×3 matrices, descending eigenvalues. */
export function symmetricEigen(input: M3) {
  const a = input.map((r) => [...r]),
    v = I3.map((r) => [...r]);
  for (let step = 0; step < 60; step++) {
    let p = 0,
      q = 1;
    for (let i = 0; i < 3; i++)
      for (let j = i + 1; j < 3; j++)
        if (Math.abs(a[i][j]) > Math.abs(a[p][q])) {
          p = i;
          q = j;
        }
    if (Math.abs(a[p][q]) < 1e-12) break;
    const theta = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]),
      c = Math.cos(theta),
      s = Math.sin(theta);
    const app = a[p][p],
      aqq = a[q][q],
      apq = a[p][q];
    for (let k = 0; k < 3; k++)
      if (k !== p && k !== q) {
        const kp = a[k][p],
          kq = a[k][q];
        a[k][p] = a[p][k] = c * kp - s * kq;
        a[k][q] = a[q][k] = s * kp + c * kq;
      }
    a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    a[p][q] = a[q][p] = 0;
    for (let k = 0; k < 3; k++) {
      const kp = v[k][p],
        kq = v[k][q];
      v[k][p] = c * kp - s * kq;
      v[k][q] = s * kp + c * kq;
    }
  }
  const order = [0, 1, 2].sort((i, j) => a[j][j] - a[i][i]);
  return {
    values: order.map((i) => a[i][i]) as unknown as V3,
    vectors: order.map((i) => v.map((r) => r[i])) as unknown as M3,
  };
}
/** vectors above are returned as a list of eigenvectors, not matrix rows. */
export function svd3(a: M3) {
  const eig = symmetricEigen(mm(transpose(a), a));
  const s = eig.values.map((x) => Math.sqrt(Math.max(0, x))) as unknown as V3;
  const columns: V3[] = [];
  for (let i = 0; i < 3; i++) {
    let u = s[i] > 1e-8 ? times(mv(a, eig.vectors[i]), 1 / s[i]) : null;
    if (u)
      for (const prev of columns) u = minus(u, times(prev, inner(u, prev)));
    if (!u || length(u) < 1e-8) {
      const candidates = I3.map((axis) =>
        columns.reduce(
          (r, prev) => minus(r, times(prev, inner(r, prev))),
          axis,
        ),
      );
      u = candidates.sort((x, y) => length(y) - length(x))[0];
    }
    columns.push(unit(u)!);
  }
  return { u: transpose(columns as unknown as M3), s, vt: eig.vectors };
}
export function covariance(points: readonly V3[]) {
  const mean = points.length
    ? times(points.reduce(plus, [0, 0, 0]), 1 / points.length)
    : ([0, 0, 0] as V3);
  const centered = points.map((p) => minus(p, mean));
  const matrix = I3.map((_, i) =>
    I3.map(
      (_, j) =>
        centered.reduce((sum, p) => sum + p[i] * p[j], 0) /
        Math.max(1, points.length - 1),
    ),
  ) as unknown as M3;
  return { mean, centered, matrix, ...symmetricEigen(matrix) };
}
export function eigen2(a: number, b: number, c: number, d: number) {
  const discriminant = (a + d) ** 2 - 4 * (a * d - b * c);
  if (discriminant < -1e-10) return { complex: true, pairs: [] };
  const values = [
    (a + d + Math.sqrt(Math.max(0, discriminant))) / 2,
    (a + d - Math.sqrt(Math.max(0, discriminant))) / 2,
  ];
  return {
    complex: false,
    pairs: values.map((value, i) => {
      const choices: V3[] = [
        [b, value - a, 0],
        [value - d, c, 0],
      ];
      const candidate = choices.sort((x, y) => length(y) - length(x))[0];
      return { value, vector: unit(candidate) ?? I3[i] };
    }),
  };
}
export function neighbours(points: readonly V3[], center: V3, k = 6) {
  return points
    .map((point, index) => ({
      point,
      index,
      distance: length(minus(point, center)),
    }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, k);
}
export function cloud(
  kind: "line" | "sheet" | "volume" = "sheet",
  noise = 0.08,
): V3[] {
  return Array.from({ length: 160 }, (_, i) => {
    const x = (((i * 37) % 157) / 157) * 4 - 2,
      y = (((i * 71) % 163) / 163) * 4 - 2,
      z = (((i * 97) % 167) / 167) * 4 - 2;
    return kind === "line"
      ? [x, 0.5 * x + noise * y, 0.3 * x + noise * z]
      : kind === "sheet"
        ? [x, 0.6 * x + y, 0.25 * x + 0.12 * y + noise * z]
        : [x, y, z];
  });
}
export function spherePoints(radius = 1): V3[] {
  return Array.from({ length: 240 }, (_, i) => {
    const z = 1 - (2 * (i + 0.5)) / 240,
      a = i * 2.3999632297,
      r = Math.sqrt(1 - z * z);
    return [radius * r * Math.cos(a), radius * r * Math.sin(a), radius * z];
  });
}
