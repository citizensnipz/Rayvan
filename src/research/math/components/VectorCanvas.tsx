import { useId, useState } from "react";
import type { ReactNode, PointerEvent } from "react";
import type { Vec2 } from "../types";
import { fmt } from "../math";
export const colors = {
  original: "#38c6cc",
  secondary: "#b299ff",
  result: "#d8ff75",
  gradient: "#ef7b86",
  muted: "#7e8a9c",
};
export interface CanvasMap {
  x: (x: number) => number;
  y: (y: number) => number;
  unit: number;
  extent: number;
}
export function CoordinateGrid({ map }: { map: CanvasMap }) {
  const step = Math.max(1, Math.ceil(map.extent / 8));
  const ticks = Array.from(
    { length: Math.floor(map.extent / step) * 2 + 1 },
    (_, i) => (i - Math.floor(map.extent / step)) * step,
  );
  return (
    <g>
      {ticks.map((n) => (
        <g key={n}>
          <path
            d={`M ${map.x(n)} 28 V 472 M 28 ${map.y(n)} H 472`}
            stroke={n === 0 ? "#536071" : "#232e3b"}
            strokeWidth={n === 0 ? 1.5 : 1}
          />
          {n !== 0 && (
            <>
              <text x={map.x(n)} y={map.y(0) + 15} className="math-tick">
                {fmt(n)}
              </text>
              <text x={map.x(0) + 7} y={map.y(n) - 5} className="math-tick">
                {fmt(n)}
              </text>
            </>
          )}
        </g>
      ))}
      <text x="481" y="245" className="math-tick">
        x
      </text>
      <text x="257" y="17" className="math-tick">
        y
      </text>
    </g>
  );
}
export function VectorCanvas({
  title,
  extent = 6,
  children,
  onPoint,
}: {
  title: string;
  extent?: number;
  children: (map: CanvasMap) => ReactNode;
  onPoint?: (v: Vec2) => void;
}) {
  const [dragExtent, setDragExtent] = useState<number | null>(null);
  extent = dragExtent ?? extent;
  const map: CanvasMap = {
    x: (x) => 250 + (x * 218) / extent,
    y: (y) => 250 - (y * 218) / extent,
    unit: 218 / extent,
    extent,
  };
  const move = (e: PointerEvent<SVGSVGElement>) => {
    const ctm = e.currentTarget.getScreenCTM();
    if (!ctm || !onPoint) return;
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse());
    onPoint([
      Math.max(-5, Math.min(5, (p.x - 250) / map.unit)),
      Math.max(-5, Math.min(5, (250 - p.y) / map.unit)),
    ]);
  };
  return (
    <svg
      className={`math-canvas ${onPoint ? "math-movable" : ""}`}
      viewBox="0 0 500 500"
      role="img"
      aria-label={title}
      onPointerDown={
        onPoint
          ? (e) => {
              setDragExtent(extent);
              e.currentTarget.setPointerCapture(e.pointerId);
              move(e);
            }
          : undefined
      }
      onPointerUp={() => setDragExtent(null)}
      onPointerCancel={() => setDragExtent(null)}
      onPointerMove={
        onPoint
          ? (e) => {
              if (e.buttons === 1) move(e);
            }
          : undefined
      }
    >
      <title>{title}</title>
      <CoordinateGrid map={map} />
      {children(map)}
    </svg>
  );
}
export function Arrow({
  map,
  to,
  from = [0, 0],
  color,
  label,
  dashed = false,
  labelOffset = [8, -9],
}: {
  map: CanvasMap;
  to: Vec2;
  from?: Vec2;
  color: string;
  label: string;
  dashed?: boolean;
  labelOffset?: Vec2;
}) {
  const id = useId().replace(/:/g, "");
  const zero = Math.hypot(to[0] - from[0], to[1] - from[1]) < 1e-9;
  return (
    <g>
      <defs>
        <marker
          id={id}
          markerWidth="8"
          markerHeight="8"
          refX="6"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M0,0 L0,6 L6,3 z" fill={color} />
        </marker>
      </defs>
      {zero ? (
        <circle cx={map.x(to[0])} cy={map.y(to[1])} r="4" fill={color} />
      ) : (
        <line
          x1={map.x(from[0])}
          y1={map.y(from[1])}
          x2={map.x(to[0])}
          y2={map.y(to[1])}
          stroke={color}
          strokeWidth="2.5"
          strokeDasharray={dashed ? "6 4" : undefined}
          markerEnd={`url(#${id})`}
        />
      )}
      <text
        x={map.x(to[0]) + labelOffset[0]}
        y={map.y(to[1]) + labelOffset[1]}
        fill={color}
        className="math-vector-label"
      >
        {label}
      </text>
    </g>
  );
}
export function Legend({
  items,
}: {
  items: readonly (readonly [string, string])[];
}) {
  return (
    <div className="math-legend">
      {items.map(([label, color]) => (
        <span key={label}>
          <i style={{ background: color }} />
          {label}
        </span>
      ))}
    </div>
  );
}
export function fitExtent(points: readonly Vec2[], minimum = 6) {
  return Math.max(
    minimum,
    ...points.flatMap((p) => p.map((n) => Math.abs(n) * 1.25)),
  );
}
