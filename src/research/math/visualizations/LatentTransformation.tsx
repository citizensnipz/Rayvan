import { useState } from "react";
import type { LatentState, Trajectory, Vec2 } from "../types";
import { coords, norm, trajectory } from "../math";
import { VectorControls } from "../components/ParameterControls";
import { VisualizationShell } from "../components/VisualizationShell";
import { TimelinePlayer } from "../components/TimelinePlayer";
import {
  Arrow,
  colors,
  fitExtent,
  Legend,
  VectorCanvas,
} from "../components/VectorCanvas";
/** Pure renderer: future adapters supply projected state frames and axis metadata here. */
export function LatentTrajectoryCanvas({
  data,
  index,
}: {
  data: Trajectory<Vec2>;
  index: number;
}) {
  const points = data.frames.map((f) => f.state),
    z = points[index],
    next = points[index + 1];
  return (
    <>
      <Legend
        items={[
          ["Current z", colors.original],
          ["Expert delta", colors.secondary],
          ["Updated z_next", colors.result],
        ]}
      />
      <VectorCanvas
        title={`Latent trajectory, selected transition ${index}`}
        extent={fitExtent(points)}
      >
        {(map) => (
          <>
            <polyline
              points={points
                .map((p) => `${map.x(p[0])},${map.y(p[1])}`)
                .join(" ")}
              fill="none"
              stroke={colors.muted}
              strokeDasharray="5 4"
            />
            {points.map((p, i) => (
              <g key={i}>
                <circle
                  cx={map.x(p[0])}
                  cy={map.y(p[1])}
                  r="3"
                  fill={i <= index ? colors.original : colors.muted}
                />
                <text
                  x={map.x(p[0]) + 7}
                  y={map.y(p[1]) + 16}
                  className="math-tick"
                >
                  z{i}
                </text>
              </g>
            ))}
            {z && (
              <Arrow
                map={map}
                to={z}
                color={colors.original}
                label={`z${index}`}
              />
            )}
            {next && (
              <>
                <Arrow
                  map={map}
                  to={next}
                  color={colors.result}
                  label={`z${index + 1}`}
                  dashed
                />
                <Arrow
                  map={map}
                  from={z}
                  to={next}
                  color={colors.secondary}
                  label="Δ(z)"
                />
              </>
            )}
          </>
        )}
      </VectorCanvas>
      {data.projection && (
        <p className="math-help">
          Axes: {data.projection.axisLabels.join(" / ")} ·{" "}
          {data.projection.method} projection from{" "}
          {data.projection.originalDimensions} dimensions.
        </p>
      )}
    </>
  );
}
export function LatentTransformation() {
  const [state, setState] = useState<LatentState>({
    initial: [-2, -1],
    deltas: [
      [1, 2],
      [2, -0.5],
      [-0.5, 1.5],
    ],
  });
  const [index, setIndex] = useState(0);
  const points = trajectory(state.initial, state.deltas),
    z = points[index],
    delta = state.deltas[index],
    next = points[index + 1];
  const data: Trajectory<Vec2> = {
    source: "manual",
    frames: points.map((state, step) => ({ state, step })),
  };
  return (
    <VisualizationShell
      title="Latent transformation"
      subtitle="A shared state changes through sequential expert updates."
      equation={`z_next = z + Δ(z) = ${coords(z)} + ${coords(delta)} = ${coords(next)}`}
      controls={
        <>
          <VectorControls
            label="Initial z₀"
            value={state.initial}
            onChange={(initial) => setState({ ...state, initial })}
          />
          <VectorControls
            label={`Delta step ${index}`}
            value={delta}
            onChange={(delta) =>
              setState({
                ...state,
                deltas: state.deltas.map((d, i) => (i === index ? delta : d)),
              })
            }
          />
          <p className="math-help">
            Editing a delta recomputes every later state. Each selected frame
            represents one transition, including its resulting state.
          </p>
        </>
      }
      values={[
        [`z${index}`, coords(z)],
        ["Δ(z)", coords(delta)],
        [`z${index + 1}`, coords(next)],
        ["Update magnitude", norm(delta).toFixed(4)],
        ["Transitions", state.deltas.length],
      ]}
      interpretation={`The expert moves the current latent state from ${coords(z)} to ${coords(next)} by adding ${coords(delta)}. The purple arrow starts at z; the green arrow locates the resulting state relative to the origin. ${norm(delta) === 0 ? "A zero delta leaves the state unchanged." : ""} The dashed path shows all sequential updates.`}
      meaning="This illustrates a residual latent update. Deltas here are manually supplied examples, not an expert prediction or evidence of learned routing. Future projected trajectories can use the same renderer."
      timeline={
        <TimelinePlayer
          frames={state.deltas.map((state, step) => ({ state, step }))}
          index={index}
          onSelect={setIndex}
          onAppend={() => {
            setState({ ...state, deltas: [...state.deltas, [1, 0]] });
            setIndex(state.deltas.length);
          }}
          onRemove={() => {
            setState({
              ...state,
              deltas: state.deltas.filter((_, i) => i !== index),
            });
            setIndex(Math.max(0, index - 1));
          }}
          onGenerate={() => {
            setState({
              initial: [-3, -2],
              deltas: [
                [1, 2],
                [1, 1],
                [2, -1],
                [-1, -2],
                [-2, 0.5],
              ],
            });
            setIndex(0);
          }}
        />
      }
    >
      <LatentTrajectoryCanvas data={data} index={index} />
    </VisualizationShell>
  );
}
