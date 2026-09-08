import { length, minus, type V3 } from "./geometry";
export type Landscape =
  "bowl" | "valley" | "saddle" | "multi-basin" | "plane" | "wave";
export function surface(kind: Landscape, x: number, y: number) {
  if (kind === "plane") return 0;
  if (kind === "bowl") return 0.25 * (x * x + y * y);
  if (kind === "valley") return 0.1 * x * x + y * y;
  if (kind === "saddle") return 0.25 * (x * x - y * y);
  if (kind === "wave") return 0.6 * Math.sin(x) * Math.cos(y);
  return 0.15 * (x * x + y * y) + 0.55 * (Math.cos(2 * x) + Math.cos(2 * y));
}
export function derivatives(kind: Landscape, x: number, y: number) {
  if (kind === "plane") return { gx: 0, gy: 0, xx: 0, xy: 0, yy: 0 };
  if (kind === "bowl")
    return { gx: 0.5 * x, gy: 0.5 * y, xx: 0.5, xy: 0, yy: 0.5 };
  if (kind === "valley")
    return { gx: 0.2 * x, gy: 2 * y, xx: 0.2, xy: 0, yy: 2 };
  if (kind === "saddle")
    return { gx: 0.5 * x, gy: -0.5 * y, xx: 0.5, xy: 0, yy: -0.5 };
  if (kind === "wave")
    return {
      gx: 0.6 * Math.cos(x) * Math.cos(y),
      gy: -0.6 * Math.sin(x) * Math.sin(y),
      xx: -0.6 * Math.sin(x) * Math.cos(y),
      xy: -0.6 * Math.cos(x) * Math.sin(y),
      yy: -0.6 * Math.sin(x) * Math.cos(y),
    };
  return {
    gx: 0.3 * x - 1.1 * Math.sin(2 * x),
    gy: 0.3 * y - 1.1 * Math.sin(2 * y),
    xx: 0.3 - 2.2 * Math.cos(2 * x),
    xy: 0,
    yy: 0.3 - 2.2 * Math.cos(2 * y),
  };
}
export function optimization(
  kind: Landscape,
  start: V3,
  rate: number,
  steps: number,
) {
  let [x, y] = start;
  const points: V3[] = [[x, y, surface(kind, x, y)]];
  let stopped = false;
  for (let i = 0; i < steps; i++) {
    const g = derivatives(kind, x, y);
    const nx = x - rate * g.gx,
      ny = y - rate * g.gy;
    if (!Number.isFinite(nx + ny) || Math.abs(nx) > 4 || Math.abs(ny) > 4) {
      stopped = true;
      break;
    }
    x = nx;
    y = ny;
    points.push([x, y, surface(kind, x, y)]);
  }
  return { points, stopped };
}
export function curvatures(kind: Landscape, x: number, y: number) {
  const { gx, gy, xx, xy, yy } = derivatives(kind, x, y),
    w = 1 + gx * gx + gy * gy;
  const gaussian = (xx * yy - xy * xy) / (w * w);
  const mean =
    ((1 + gy * gy) * xx - 2 * gx * gy * xy + (1 + gx * gx) * yy) /
    (2 * w ** 1.5);
  const root = Math.sqrt(Math.max(0, mean * mean - gaussian));
  return { gaussian, mean, principal: [mean + root, mean - root] };
}
export function manifoldPath(kind: Landscape, a: V3, b: V3) {
  const points: V3[] = Array.from({ length: 61 }, (_, i) => {
    const t = i / 60,
      x = a[0] * (1 - t) + b[0] * t,
      y = a[1] * (1 - t) + b[1] * t;
    return [x, y, surface(kind, x, y)];
  });
  return {
    points,
    distance: points
      .slice(1)
      .reduce((s, p, i) => s + length(minus(p, points[i])), 0),
  };
}
