import { lazy, Suspense, useState } from "react";
import { Arrow, VectorCanvas, fitExtent } from "../components/VectorCanvas";
import type { GeometryScene } from "./scene";
const ThreeDScene = lazy(() => import("./ThreeDScene"));
export function GeometryView({
  scene,
  mode: controlled,
  onMode,
  caption = "Toy example: genuinely 3D coordinates. 2D view drops z; it does not preserve all distances.",
}: {
  scene: GeometryScene;
  mode?: "2D" | "3D";
  onMode?: (mode: "2D" | "3D") => void;
  caption?: string;
}) {
  const [local, setLocal] = useState<"2D" | "3D">("3D");
  const mode = controlled ?? local;
  const available =
    typeof window !== "undefined" &&
    typeof window.WebGL2RenderingContext !== "undefined";
  const points = [
    ...(scene.points?.map((p) => p.position) ?? []),
    ...(scene.arrows?.flatMap((a) => [a.from ?? [0, 0, 0], a.to]) ?? []),
    ...(scene.clouds?.flatMap((c) => c.points) ?? []),
    ...(scene.paths?.flatMap((p) => p.points) ?? []),
  ];
  const extent = fitExtent(
    points.map((p) => [p[0], p[1]]),
    4,
  );
  return (
    <>
      <div className="math-view-switch" role="group" aria-label="Geometry view">
        {(["2D", "3D"] as const).map((value) => (
          <button
            key={value}
            aria-pressed={mode === value}
            onClick={() => {
              setLocal(value);
              onMode?.(value);
            }}
          >
            {value}
          </button>
        ))}
      </div>
      <p className="math-science-note">{caption}</p>
      {mode === "3D" && available ? (
        <Suspense fallback={<p>Loading 3D renderer…</p>}>
          <ThreeDScene scene={scene} />
        </Suspense>
      ) : (
        <>
          {mode === "3D" && (
            <p role="status" className="math-help">
              WebGL2 unavailable: showing the XY projection. Numeric controls
              remain available.
            </p>
          )}
          <VectorCanvas title="XY projection of geometry" extent={extent}>
            {(map) => (
              <>
                {scene.surfaces?.map((s, i) => (
                  <g key={i}>
                    {s.indices
                      .filter((_, j) => j % 3 === 0)
                      .map((_, j) => {
                        const v = s.indices
                          .slice(j * 3, j * 3 + 3)
                          .map((k) => s.vertices[k]);
                        return (
                          <polygon
                            key={j}
                            points={v
                              .map((p) => `${map.x(p[0])},${map.y(p[1])}`)
                              .join(" ")}
                            fill={s.color ?? "#38c6cc"}
                            opacity={0.035}
                          />
                        );
                      })}
                  </g>
                ))}
                {scene.regions?.map((r, i) => (
                  <circle
                    key={i}
                    cx={map.x(r.center[0])}
                    cy={map.y(r.center[1])}
                    r={r.radius * map.unit}
                    fill={r.color}
                    fillOpacity=".04"
                    stroke={r.color}
                    strokeOpacity=".3"
                  />
                ))}
                {scene.clouds?.flatMap((c, i) =>
                  c.points.map((p, j) => (
                    <circle
                      key={`${i}-${j}`}
                      cx={map.x(p[0])}
                      cy={map.y(p[1])}
                      r="2.5"
                      fill={c.color}
                    />
                  )),
                )}
                {scene.paths?.map((p, i) => (
                  <polyline
                    key={i}
                    points={p.points
                      .map((v) => `${map.x(v[0])},${map.y(v[1])}`)
                      .join(" ")}
                    fill="none"
                    stroke={p.color ?? "#d8ff75"}
                    strokeDasharray={p.dashed ? "4 4" : undefined}
                  />
                ))}
                {scene.arrows?.map((a, i) => (
                  <Arrow
                    key={i}
                    map={map}
                    to={[a.to[0], a.to[1]]}
                    from={a.from ? [a.from[0], a.from[1]] : undefined}
                    color={a.color ?? "#d8ff75"}
                    label={a.label ?? ""}
                  />
                ))}
                {scene.points?.map((p, i) => (
                  <g key={i}>
                    <circle
                      cx={map.x(p.position[0])}
                      cy={map.y(p.position[1])}
                      r="5"
                      fill={p.color ?? "#d8ff75"}
                    />
                    <text
                      x={map.x(p.position[0]) + 8}
                      y={map.y(p.position[1]) - 8}
                      className="math-tick"
                    >
                      {p.label}
                    </text>
                  </g>
                ))}
              </>
            )}
          </VectorCanvas>
        </>
      )}
    </>
  );
}
