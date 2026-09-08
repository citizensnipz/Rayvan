import type { VectorState, Vec2 } from "../types";
import {
  add,
  coords,
  cosine,
  distance,
  dot,
  fmt,
  norm,
  projection,
  subtract,
} from "../math";
import {
  VectorControls,
  ParameterControl,
} from "../components/ParameterControls";
import { VisualizationShell } from "../components/VisualizationShell";
import { useSequence } from "../components/TimelinePlayer";
import {
  Arrow,
  colors,
  fitExtent,
  Legend,
  VectorCanvas,
} from "../components/VectorCanvas";
type Mode = "vectors" | "norm" | "dot" | "cosine" | "distance" | "projection";
const titles: Record<Mode, string> = {
  vectors: "Vector basics",
  norm: "Vector magnitude / norm",
  dot: "Dot product",
  cosine: "Cosine similarity",
  distance: "Euclidean distance",
  projection: "Vector projection",
};
const meanings: Record<Mode, string> = {
  vectors:
    "A latent state is a list of coordinates. This arrow shows two of them; its length and direction describe different properties.",
  norm: "Norms quantify the size of activations, updates, and gradients. A bigger norm does not automatically mean more useful information.",
  dot: "Attention scores commonly use dot products: both alignment and vector size affect the score.",
  cosine:
    "Normalized embedding comparisons ignore magnitude. Alignment can represent similarity when the embedding has learned a meaningful geometry.",
  distance:
    "Distances can compare states to expert prototypes, but their meaning depends on the representation and its scale.",
  projection:
    "Projection isolates the part of a state or update that lies along a chosen direction. The residual contains the perpendicular component.",
};
export function VectorVisualization({ mode }: { mode: Mode }) {
  const { state, update, player } = useSequence<VectorState>(
    { u: [3, 2], v: [2, -1] },
    () =>
      [0, 45, 90, 135, 180].map((angle) => ({
        u: [3, 0],
        v: [
          3 * Math.cos((angle * Math.PI) / 180),
          3 * Math.sin((angle * Math.PI) / 180),
        ],
      })),
  );
  const { u, v } = state,
    single = mode === "vectors" || mode === "norm";
  const product = dot(u, v),
    similarity = cosine(u, v),
    projected = projection(u, v);
  const angle =
    similarity === null ? null : (Math.acos(similarity) * 180) / Math.PI;
  const alignment =
    similarity === null
      ? "Direction and angle are undefined because at least one vector is zero."
      : Math.abs(similarity) < 1e-8
        ? "The vectors are perpendicular: their dot product is zero."
        : similarity >= 0.8
          ? "The vectors are strongly aligned."
          : similarity > 0.2
            ? "The vectors are partly aligned."
            : similarity >= -0.2
              ? "The vectors have weak directional alignment."
              : similarity > -0.8
                ? "The vectors point partly against each other."
                : "The vectors are strongly opposed.";
  const equation = single
    ? `‖u‖ = √(uₓ² + uᵧ²) = √((${fmt(u[0])})² + (${fmt(u[1])})²) = ${fmt(norm(u))}`
    : mode === "dot"
      ? `u · v = uₓvₓ + uᵧvᵧ = (${fmt(u[0])})(${fmt(v[0])}) + (${fmt(u[1])})(${fmt(v[1])}) = ${fmt(product)}`
      : mode === "cosine"
        ? `cos θ = (u · v) / (‖u‖ ‖v‖) = ${fmt(product)} / (${fmt(norm(u))} × ${fmt(norm(v))}) = ${similarity === null ? "undefined" : fmt(similarity)}`
        : mode === "distance"
          ? `d(u, v) = ‖u − v‖ = ‖${coords(subtract(u, v))}‖ = ${fmt(distance(u, v))}`
          : `projᵥ(u) = [(u · v) / (v · v)] v = ${projected ? `${fmt(product / dot(v, v))} × ${coords(v)} = ${coords(projected)}` : "undefined (v is zero)"}`;
  const values: [string, string][] = single
    ? [
        ["u", coords(u)],
        ["x² contribution", fmt(u[0] ** 2)],
        ["y² contribution", fmt(u[1] ** 2)],
        ["‖u‖", fmt(norm(u))],
      ]
    : [
        ["u", coords(u)],
        ["v", coords(v)],
        ["u · v", fmt(product)],
        ["‖u‖", fmt(norm(u))],
        ["‖v‖", fmt(norm(v))],
        [
          "Cosine similarity",
          similarity === null ? "Undefined" : fmt(similarity),
        ],
        ["Angle θ", angle === null ? "Undefined" : `${fmt(angle)}°`],
      ];
  if (mode === "distance") values.push(["Distance", fmt(distance(u, v))]);
  if (mode === "projection")
    values.push(
      ["Projection", projected ? coords(projected) : "Undefined"],
      [
        "Residual u − projection",
        projected ? coords(subtract(u, projected)) : "Undefined",
      ],
    );
  const polar = (label: string, vector: Vec2, set: (v: Vec2) => void) => (
    <fieldset className="math-vector-controls">
      <legend>{label} direction / magnitude</legend>
      <ParameterControl
        label={`${label} angle °`}
        min={-180}
        max={180}
        step={1}
        value={(Math.atan2(vector[1], vector[0]) * 180) / Math.PI}
        onChange={(a) =>
          set([
            norm(vector) * Math.cos((a * Math.PI) / 180),
            norm(vector) * Math.sin((a * Math.PI) / 180),
          ])
        }
      />
      <ParameterControl
        label={`${label} magnitude`}
        min={0}
        max={5}
        value={norm(vector)}
        onChange={(r) => {
          const a = Math.atan2(vector[1], vector[0]);
          set([r * Math.cos(a), r * Math.sin(a)]);
        }}
      />
    </fieldset>
  );
  return (
    <VisualizationShell
      title={titles[mode]}
      subtitle={
        single
          ? "Coordinates define direction; the norm measures length."
          : "Change each coordinate and watch the relationship update."
      }
      equation={equation}
      controls={
        <>
          {mode !== "cosine" && (
            <VectorControls
              label="u"
              value={u}
              onChange={(u) => update({ ...state, u })}
            />
          )}
          {!single && mode !== "cosine" && (
            <VectorControls
              label="v"
              value={v}
              onChange={(v) => update({ ...state, v })}
            />
          )}
          {mode === "cosine" && (
            <>
              {polar("u", u, (u) => update({ ...state, u }))}
              {polar("v", v, (v) => update({ ...state, v }))}
            </>
          )}
        </>
      }
      values={values}
      meaning={meanings[mode]}
      interpretation={
        single
          ? `Moving ${fmt(u[0])} along x and ${fmt(u[1])} along y places u at ${coords(u)}. Its distance from the origin is ${fmt(norm(u))}.${norm(u) === 0 ? " The zero vector has no direction." : ""}`
          : mode === "projection"
            ? projected
              ? `The green arrow is u’s shadow along v. The dashed residual meets that direction at a right angle. A negative coefficient places the projection opposite v.`
              : "A zero vector defines no direction to project onto. Choose a nonzero v."
            : mode === "distance"
              ? `The connecting segment has length ${fmt(distance(u, v))}. Unlike cosine similarity, this changes when vector magnitude changes.`
              : `${alignment} ${mode === "dot" ? "The shaded signed area is ‖v‖ times u’s signed length along v; it equals the dot product. Scaling either vector changes this value." : "Scaling a nonzero vector changes its dot product but leaves cosine similarity unchanged."}`
      }
      timeline={player}
    >
      <Legend
        items={
          single
            ? [["u", colors.original]]
            : [
                ["u", colors.original],
                ["v", colors.secondary],
                ...(mode === "projection"
                  ? [["projection", colors.result] as const]
                  : []),
              ]
        }
      />
      <VectorCanvas
        title={`${titles[mode]}: u ${coords(u)}${single ? "" : `, v ${coords(v)}`}`}
        extent={fitExtent([
          u,
          ...(!single ? [v] : []),
          ...(projected && mode === "projection" ? [projected] : []),
          ...(projected && mode === "dot"
            ? [projected, [-v[1], v[0]] as Vec2, add(projected, [-v[1], v[0]])]
            : []),
        ])}
      >
        {(map) => (
          <>
            {mode === "dot" &&
              projected &&
              (() => {
                const offset: Vec2 = [-v[1], v[0]];
                const corners: Vec2[] = [
                  [0, 0],
                  projected,
                  [projected[0] + offset[0], projected[1] + offset[1]],
                  offset,
                ];
                return (
                  <polygon
                    points={corners
                      .map((p) => `${map.x(p[0])},${map.y(p[1])}`)
                      .join(" ")}
                    fill={product < 0 ? colors.gradient : colors.result}
                    opacity=".13"
                  />
                );
              })()}
            {single && (
              <path
                d={`M ${map.x(0)} ${map.y(0)} H ${map.x(u[0])} V ${map.y(u[1])}`}
                stroke={colors.muted}
                fill="none"
                strokeDasharray="5 4"
              />
            )}
            {!single &&
              similarity !== null &&
              (() => {
                const start = Math.atan2(u[1], u[0]);
                let sweep = Math.atan2(v[1], v[0]) - start;
                while (sweep > Math.PI) sweep -= 2 * Math.PI;
                while (sweep < -Math.PI) sweep += 2 * Math.PI;
                const radius = 0.8 * map.unit;
                return (
                  <g>
                    <path
                      d={`M ${250 + radius * Math.cos(start)} ${250 - radius * Math.sin(start)} A ${radius} ${radius} 0 0 ${sweep >= 0 ? 0 : 1} ${250 + radius * Math.cos(start + sweep)} ${250 - radius * Math.sin(start + sweep)}`}
                      stroke={colors.muted}
                      fill="none"
                    />
                    <text
                      x={250 + (radius + 18) * Math.cos(start + sweep / 2)}
                      y={250 - (radius + 18) * Math.sin(start + sweep / 2)}
                      className="math-tick"
                    >
                      {fmt(angle!)}°
                    </text>
                  </g>
                );
              })()}
            <Arrow map={map} to={u} color={colors.original} label="u" />
            {!single && (
              <Arrow
                map={map}
                to={v}
                color={colors.secondary}
                label="v"
                labelOffset={[8, 17]}
              />
            )}
            {mode === "distance" && (
              <Arrow
                map={map}
                from={v}
                to={u}
                color={colors.result}
                label="u − v"
                dashed
              />
            )}
            {mode === "projection" && projected && (
              <>
                <Arrow
                  map={map}
                  to={projected}
                  color={colors.result}
                  label="projᵥ(u)"
                />
                <Arrow
                  map={map}
                  from={projected}
                  to={u}
                  color={colors.muted}
                  label="residual"
                  dashed
                />
              </>
            )}
          </>
        )}
      </VectorCanvas>
    </VisualizationShell>
  );
}
export const VectorBasics = () => <VectorVisualization mode="vectors" />;
export const DotProduct = () => <VectorVisualization mode="dot" />;
export const CosineSimilarity = () => <VectorVisualization mode="cosine" />;
export const VectorProjection = () => <VectorVisualization mode="projection" />;
