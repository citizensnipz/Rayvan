import type { V3 } from "../core/geometry";
export interface ScenePoint {
  position: V3;
  color?: string;
  label?: string;
  radius?: number;
  onMove?: (point: V3) => void;
}
export interface SceneArrow {
  from?: V3;
  to: V3;
  color?: string;
  label?: string;
}
export interface ScenePath {
  points: readonly V3[];
  color?: string;
  dashed?: boolean;
}
export interface SceneSurface {
  vertices: readonly V3[];
  indices: readonly number[];
  color?: string;
  opacity?: number;
}
export interface GeometryScene {
  points?: readonly ScenePoint[];
  arrows?: readonly SceneArrow[];
  paths?: readonly ScenePath[];
  clouds?: readonly { points: readonly V3[]; color: string; size?: number }[];
  surfaces?: readonly SceneSurface[];
  regions?: readonly { center: V3; radius: number; color: string }[];
}
export function surfaceMesh(
  fn: (x: number, y: number) => number,
  extent = 3,
  resolution = 32,
): SceneSurface {
  const vertices: V3[] = [],
    indices: number[] = [];
  for (let y = 0; y <= resolution; y++)
    for (let x = 0; x <= resolution; x++) {
      const px = -extent + (2 * extent * x) / resolution,
        py = -extent + (2 * extent * y) / resolution;
      vertices.push([px, py, fn(px, py)]);
    }
  for (let y = 0; y < resolution; y++)
    for (let x = 0; x < resolution; x++) {
      const a = y * (resolution + 1) + x,
        b = a + 1,
        c = a + resolution + 1,
        d = c + 1;
      indices.push(a, b, c, b, d, c);
    }
  return { vertices, indices, color: "#38c6cc", opacity: 0.55 };
}
