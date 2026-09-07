import { useState } from "react";
import { VisualizationShell } from "../../components/VisualizationShell";
import { Vector3Controls } from "../../components/SpatialControls";
import { TimelinePlayer } from "../../components/TimelinePlayer";
import { GeometryView } from "../../renderers/GeometryView";
import { f3, length, plus, type V3 } from "../../core/geometry";
export function LatentLab3D() {
  const [initial, setInitial] = useState<V3>([-2, -1, 0]),
    [deltas, setDeltas] = useState<V3[]>([
      [1, 1, 0.5],
      [1, -0.5, 1],
      [-0.5, 1, -0.5],
    ]),
    [index, setIndex] = useState(0);
  const points = deltas.reduce<V3[]>(
    (points, delta) => [...points, plus(points.at(-1)!, delta)],
    [initial],
  );
  return (
    <VisualizationShell
      title="3D latent transformation"
      context="TOY LATENT TRAJECTORY / NOT PROJECTED TELEMETRY"
      subtitle="Edit an earlier delta and every later state updates."
      equation={`z_next = z + Δ(z) = ${f3(points[index])} + ${f3(deltas[index])} = ${f3(points[index + 1])}`}
      controls={
        <>
          <Vector3Controls
            label="Initial latent"
            value={initial}
            onChange={setInitial}
          />
          <Vector3Controls
            label="Selected delta"
            value={deltas[index]}
            onChange={(delta) =>
              setDeltas(deltas.map((v, i) => (i === index ? delta : v)))
            }
          />
        </>
      }
      values={[
        ["Current state", f3(points[index])],
        ["Delta", f3(deltas[index])],
        ["Next state", f3(points[index + 1])],
        ["Update magnitude", length(deltas[index]).toFixed(3)],
      ]}
      interpretation="The purple arrow starts at the current state and ends at the updated state. The grey path includes future steps. These manually edited deltas are not expert predictions."
      meaning="A future projection adapter must preserve original dimensionality and projection metadata. For nonlinear projection, subtracting projected endpoints is not generally the same as projecting the original delta."
      timeline={
        <TimelinePlayer
          frames={deltas.map((state, step) => ({ state, step }))}
          index={index}
          onSelect={setIndex}
          onAppend={() => {
            setDeltas([...deltas, [1, 0, 0]]);
            setIndex(deltas.length);
          }}
          onRemove={() => {
            setDeltas(deltas.filter((_, i) => i !== index));
            setIndex(Math.max(0, index - 1));
          }}
          onGenerate={() => {
            setInitial([-2, -1, 0]);
            setDeltas([
              [1, 1, 0.5],
              [1, -0.5, 1],
              [-0.5, 1, -0.5],
            ]);
            setIndex(0);
          }}
        />
      }
    >
      <GeometryView
        scene={{
          points: [
            { position: points[index], color: "#38c6cc", label: `z${index}` },
            {
              position: points[index + 1],
              color: "#d8ff75",
              label: `z${index + 1}`,
            },
          ],
          arrows: [
            { to: points[index], color: "#38c6cc" },
            {
              from: points[index],
              to: points[index + 1],
              color: "#b299ff",
              label: "Δ(z)",
            },
          ],
          paths: [
            { points, color: "#536b84" },
            { points: points.slice(0, index + 2), color: "#d8ff75" },
          ],
        }}
      />
    </VisualizationShell>
  );
}
