import { useMemo, useState } from "react";
import { VisualizationShell } from "../../components/VisualizationShell";
import { Vector3Controls } from "../../components/SpatialControls";
import { ParameterControl } from "../../components/ParameterControls";
import { useMathStep } from "../../components/AnimatedMathStep";
import { GeometryView } from "../../renderers/GeometryView";
import { EChart } from "../../../charts/EChart";
import {
  experts,
  metricDistance,
  rankExperts,
  routeSequence,
  type Metric,
  type ToyExpert,
} from "../../core/routing";
import { f3, format, length, mix, unit, type V3 } from "../../core/geometry";
export type RoutingMode =
  "need" | "basins" | "metrics" | "sequential" | "inhibition";
const titles: Record<RoutingMode, string> = {
  need: "Computational-need embedding",
  basins: "Expert competence basins",
  metrics: "Distance metrics for routing",
  sequential: "Sequential routing trajectory",
  inhibition: "Refractory / inhibition routing",
};
export function RoutingLab({ mode = "basins" }: { mode?: RoutingMode }) {
  const [raw, setRaw] = useState<V3>([-2, 0.6, 0.4]),
    [list, setList] = useState<ToyExpert[]>(() => structuredClone(experts)),
    [metric, setMetric] = useState<Metric>("euclidean"),
    [strength, setStrength] = useState(0.3),
    [decay, setDecay] = useState(1.2),
    [steps, setSteps] = useState(12),
    [edit, setEdit] = useState(0);
  const sequential = mode === "sequential" || mode === "inhibition";
  const frames = useMemo(
    () =>
      routeSequence(
        raw,
        list,
        metric,
        steps,
        mode === "inhibition" ? strength : 0,
        decay,
      ),
    [raw, list, metric, steps, mode, strength, decay],
  );
  const { index, t, timeline } = useMathStep(
    sequential ? Math.max(1, frames.length) : 41,
    sequential
      ? "Recompute routing after each update"
      : "Normalize raw need → unit need",
    sequential ? 500 : 80,
  );
  const frame = frames[index],
    normalized = unit(raw),
    z = sequential
      ? (frame?.z ?? raw)
      : mode === "need" && normalized
        ? mix(raw, normalized, t)
        : raw;
  const ranks = sequential
      ? (frame?.ranks ?? [])
      : rankExperts(z, list, metric),
    winner = ranks.find((r) => Number.isFinite(r.score));
  const selected = list[Math.min(edit, list.length - 1)];
  const updateExpert = (patch: Partial<ToyExpert>) =>
    setList(list.map((e) => (e.id === selected.id ? { ...e, ...patch } : e)));
  return (
    <VisualizationShell
      title={titles[mode]}
      context="EMC RESEARCH HYPOTHESIS / TOY ROUTER"
      subtitle="Editable geometric scores and transformations—not the production EMC routing algorithm."
      equation={
        sequential
          ? "sᵢ = −d(z, μᵢ)²/(2wᵢ²) − hᵢ; hᵢ = strength·exp(−decay·age); z_next = Aᵢz + bᵢ = z + Δᵢ(z)"
          : mode === "need"
            ? `ẑ = z/‖z‖ = ${normalized ? f3(normalized) : "undefined (zero vector)"}`
            : "sᵢ = −d(z, μᵢ)²/(2wᵢ²); choose largest finite score"
      }
      controls={
        <>
          <Vector3Controls label="Raw state z" value={raw} onChange={setRaw} />
          <label>
            <span>Routing metric</span>
            <select
              value={metric}
              onChange={(e) => setMetric(e.target.value as Metric)}
            >
              <option value="euclidean">Euclidean</option>
              <option value="cosine">Cosine distance</option>
              <option value="mahalanobis">
                Mahalanobis (diagonal toy covariance)
              </option>
            </select>
          </label>
          <label>
            <span>Edit expert prototype</span>
            <select
              value={edit}
              onChange={(e) => setEdit(Number(e.target.value))}
            >
              {list.map((e, i) => (
                <option key={e.id} value={i}>
                  {e.id}
                </option>
              ))}
            </select>
          </label>
          <Vector3Controls
            label="Prototype μ"
            value={selected.center}
            onChange={(center) => updateExpert({ center })}
          />
          <ParameterControl
            label="Basin width w"
            min={0.2}
            max={2}
            value={selected.width}
            onChange={(width) => updateExpert({ width })}
          />
          <button
            disabled={list.length >= 6}
            onClick={() => {
              setList([
                ...list,
                {
                  ...experts[0],
                  id: `Expert ${list.length + 1}`,
                  center: [0, 0, 1.5],
                  color: ["#ffae70", "#ff7dbb", "#86a8ff"][list.length - 3],
                },
              ]);
              setEdit(list.length);
            }}
          >
            Add expert
          </button>
          {sequential && (
            <>
              <ParameterControl
                label="Trajectory steps"
                min={2}
                max={40}
                step={1}
                value={steps}
                onChange={setSteps}
              />
              {mode === "inhibition" && (
                <>
                  <ParameterControl
                    label="Inhibition strength"
                    min={0}
                    max={2}
                    value={strength}
                    onChange={setStrength}
                  />
                  <ParameterControl
                    label="Decay rate"
                    min={0.1}
                    max={3}
                    value={decay}
                    onChange={setDecay}
                  />
                </>
              )}
            </>
          )}
          <button
            onClick={() => {
              setRaw([-2, 0.6, 0.4]);
              setList(structuredClone(experts));
              setEdit(0);
            }}
          >
            Reset router example
          </button>
        </>
      }
      values={[
        ["Raw magnitude", format(length(raw))],
        ["Current state", f3(z)],
        ["Normalized raw state", normalized ? f3(normalized) : "Undefined"],
        ["Selected expert", winner?.expert.id ?? "No finite score"],
        ["Current step", index],
        ...(sequential && frame
          ? ([
              ["Expert delta", f3(frame.delta)],
              ["Next state", f3(frame.next)],
            ] as [string, string][])
          : []),
      ]}
      interpretation={
        mode === "need"
          ? "Normalization removes magnitude, keeping direction. Prototypes here are not normalized automatically; metric choice still matters. Zero need has no defined normalized direction."
          : mode === "inhibition"
            ? "The expert used on the previous step receives the full temporary penalty. Its penalty decays exponentially on later steps and resets when it runs again. It is never permanently excluded. Compare the score columns while scrubbing."
            : mode === "metrics"
              ? "The same state can prefer different experts under Euclidean and angular comparisons. Diagonal Mahalanobis uses variances (0.3, 2, 1), giving the first axis more weight. Widths also influence scores; these are scores, not calibrated probabilities."
              : sequential
                ? "At each frame, scores are recomputed from the updated state. The chosen toy affine expert produces the next point. Scrubbing shows that frame’s scores and before/after prototype distances."
                : "Spheres illustrate an influence width only. They do not assert that competence is spherical, Gaussian, or determined solely by semantic proximity."
      }
      meaning="Exploratory EMC tooling: computational need, competence prototypes and decaying inhibition are hypotheses to test. Coordinates and affine expert transformations are manually supplied toy examples; no actual routing or inference code is called."
      timeline={sequential || mode === "need" ? timeline : null}
    >
      <GeometryView
        scene={{
          points: [
            {
              position: z,
              label: "z",
              color: "#ffffff",
              onMove: sequential ? undefined : setRaw,
            },
            ...list.map((expert) => ({
              position: expert.center,
              label: expert.id,
              color: expert.color,
              onMove: (center: V3) =>
                setList(
                  list.map((e) => (e.id === expert.id ? { ...e, center } : e)),
                ),
            })),
          ],
          regions: list.map((e) => ({
            center: e.center,
            radius: e.width,
            color: e.color,
          })),
          arrows:
            frame && sequential
              ? [
                  {
                    from: frame.z,
                    to: frame.next,
                    color: winner?.expert.color,
                    label: "Δ(z)",
                  },
                ]
              : [{ to: z, label: "need", color: "#d8ff75" }],
          paths:
            sequential && frames.length
              ? [
                  {
                    points: [frames[0].z, ...frames.map((f) => f.next)],
                    color: "#536b84",
                  },
                  {
                    points: [
                      frames[0].z,
                      ...frames.slice(0, index + 1).map((f) => f.next),
                    ],
                    color: "#d8ff75",
                  },
                ]
              : [],
        }}
      />
      <div className="math-coordinate-table">
        <table>
          <thead>
            <tr>
              <th>Expert / rank</th>
              <th>Distance</th>
              <th>Base score</th>
              <th>Inhibition</th>
              <th>Final score</th>
              {sequential && <th>Distance after</th>}
            </tr>
          </thead>
          <tbody>
            {ranks.map((r, i) => (
              <tr key={r.expert.id}>
                <td>
                  {i + 1}. {r.expert.id}
                </td>
                <td>{format(r.distance)}</td>
                <td>{format(r.base)}</td>
                <td>{format(r.penalty)}</td>
                <td>{format(r.score)}</td>
                {sequential && (
                  <td>
                    {format(
                      metricDistance(frame?.next ?? z, r.expert.center, metric),
                    )}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {mode === "metrics" && (
        <div className="math-coordinate-table">
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Selected expert</th>
                <th>Best score</th>
              </tr>
            </thead>
            <tbody>
              {(["euclidean", "cosine", "mahalanobis"] as Metric[]).map((m) => {
                const best = rankExperts(z, list, m)[0];
                return (
                  <tr key={m}>
                    <td>{m}</td>
                    <td>
                      {Number.isFinite(best.score)
                        ? best.expert.id
                        : "Undefined"}
                    </td>
                    <td>{format(best.score)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {mode === "inhibition" && typeof ResizeObserver !== "undefined" && (
        <EChart
          option={{
            backgroundColor: "transparent",
            animation: false,
            grid: { left: 70, right: 20, top: 35, bottom: 45 },
            legend: { textStyle: { color: "#9aa8bb" } },
            xAxis: {
              type: "category",
              data: ranks.map((r) => r.expert.id),
              axisLabel: { color: "#9aa8bb" },
            },
            yAxis: { type: "value", axisLabel: { color: "#9aa8bb" } },
            series: [
              {
                name: "Before inhibition",
                type: "bar",
                data: ranks.map((r) =>
                  Number.isFinite(r.base) ? r.base : null,
                ),
                color: "#38c6cc",
              },
              {
                name: "After inhibition",
                type: "bar",
                data: ranks.map((r) =>
                  Number.isFinite(r.score) ? r.score : null,
                ),
                color: "#b299ff",
              },
            ],
          }}
        />
      )}
    </VisualizationShell>
  );
}
