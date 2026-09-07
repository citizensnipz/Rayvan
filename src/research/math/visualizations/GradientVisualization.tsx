import type { GradientState } from "../types";
import { add, coords, descent, fmt, gradient, loss, norm } from "../math";
import {
  ParameterControl,
  VectorControls,
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
export function GradientVisualization() {
  const { state, update, player } = useSequence<GradientState>(
    { point: [2, 2], weights: [1, 1], rate: 0.15 },
    () =>
      Array.from({ length: 7 }, (_, i) => ({
        point: [3 * 0.7 ** i, 2 * 0.7 ** i],
        weights: [1, 1],
        rate: 0.15,
      })),
  );
  const { point, weights, rate } = state,
    g = gradient(point, weights),
    next = descent(point, weights, rate),
    before = loss(point, weights),
    after = loss(next, weights);
  const extent = fitExtent([point, next, add(point, g)]);
  return (
    <VisualizationShell
      title="Gradient and gradient descent"
      subtitle="Drag on the plot to move the point; contours connect equal loss values."
      equation={`L(x, y) = ${fmt(weights[0])}x² + ${fmt(weights[1])}y²; ∇L = (2ax, 2by) = ${coords(g)}; p′ = p − η∇L = ${coords(next)}`}
      controls={
        <>
          <VectorControls
            label="Point"
            value={point}
            onChange={(point) => update({ ...state, point })}
          />
          <ParameterControl
            label="x² weight a"
            min={0.1}
            max={3}
            value={weights[0]}
            onChange={(a) => update({ ...state, weights: [a, weights[1]] })}
          />
          <ParameterControl
            label="y² weight b"
            min={0.1}
            max={3}
            value={weights[1]}
            onChange={(b) => update({ ...state, weights: [weights[0], b] })}
          />
          <ParameterControl
            label="Learning rate η"
            min={0}
            max={1.2}
            step={0.01}
            value={rate}
            onChange={(rate) => update({ ...state, rate })}
          />
        </>
      }
      values={[
        ["Point p", coords(point)],
        ["∇L components", coords(g)],
        ["‖∇L‖", fmt(norm(g))],
        ["Next point p′", coords(next)],
        ["L before", fmt(before)],
        ["L after", fmt(after)],
        ["Loss change", fmt(after - before)],
      ]}
      interpretation={`${norm(g) === 0 ? "At the minimum, the gradient is zero and the point stays put." : "The red arrow is the actual gradient, pointing uphill. The green arrow is −η∇L, pointing downhill from the same point."} ${after > before + 1e-9 ? "This learning rate overshoots: the landing point has higher loss, despite starting in a downhill direction." : after < before - 1e-9 ? `This step reduces loss by ${fmt(before - after)}.` : "This step leaves the loss unchanged."} The view automatically fits both arrows; axis ticks show the scale.`}
      meaning="Backpropagation computes loss gradients with respect to parameters. Here the two adjustable coordinates stand in for parameters; a descent update moves against their gradient."
      timeline={player}
    >
      <Legend
        items={[
          ["Point p", colors.original],
          ["∇L (uphill)", colors.gradient],
          ["−η∇L (descent)", colors.result],
        ]}
      />
      <VectorCanvas
        title="Editable quadratic loss contours with gradient and descent step"
        extent={extent}
        onPoint={(point) => update({ ...state, point })}
      >
        {(map) => (
          <>
            {Array.from(
              { length: 9 },
              (_, i) => ((9 - i) * extent ** 2) / 5,
            ).map((level) => (
              <g key={level}>
                <ellipse
                  cx="250"
                  cy="250"
                  rx={Math.sqrt(level / weights[0]) * map.unit}
                  ry={Math.sqrt(level / weights[1]) * map.unit}
                  stroke="#456275"
                  fill="#38c6cc"
                  fillOpacity=".018"
                />
                <text
                  x="255"
                  y={250 - Math.sqrt(level / weights[1]) * map.unit}
                  className="math-tick"
                >
                  L={fmt(level)}
                </text>
              </g>
            ))}
            <Arrow
              map={map}
              from={point}
              to={add(point, g)}
              color={colors.gradient}
              label="∇L"
            />
            <Arrow
              map={map}
              from={point}
              to={next}
              color={colors.result}
              label="p′"
            />
            <circle
              cx={map.x(point[0])}
              cy={map.y(point[1])}
              r="6"
              fill={colors.original}
            />
            <text
              x={map.x(point[0]) + 10}
              y={map.y(point[1]) + 17}
              className="math-vector-label"
              fill={colors.original}
            >
              p
            </text>
          </>
        )}
      </VectorCanvas>
    </VisualizationShell>
  );
}
