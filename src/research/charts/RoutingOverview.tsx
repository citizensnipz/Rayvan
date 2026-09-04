import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import { EChart } from "./EChart";
import type { ResearchEvent } from "../types";

type RoutingField = [name: string, field: string, color: string];
const REGRET_FIELDS: RoutingField[] = [
  ["Mean regret", "mean_routing_regret", "#ef7b86"],
  ["Median regret", "median_routing_regret", "#f2d276"],
  ["P90 regret", "p90_routing_regret", "#8e69ff"],
];
const MARGIN_FIELDS: RoutingField[] = [
  ["Best vs runner-up", "mean_geometric_margin", "#8e69ff"],
];

export function RoutingOverview({ events }: { events: ResearchEvent[] }) {
  const option = useMemo<EChartsOption>(() => {
    const latest = events.filter((event) => event.type === "routing_metrics").at(-1);
    const names = (latest?.expert_names as string[] | undefined) ?? [];
    const utilization = (latest?.utilization as number[] | undefined) ?? [];
    const probabilities = (latest?.mean_probabilities as number[] | undefined) ?? [];
    return {
      title: { text: "Latest routing balance", left: 14, top: 10, textStyle: { color: "#e8edf5", fontSize: 13 } },
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
      legend: { top: 10, right: 12, textStyle: { color: "#93a0b2" } },
      grid: { left: 84, right: 22, top: 48, bottom: 30 },
      xAxis: { type: "value", max: 1, axisLabel: { color: "#77859a", formatter: "{value}" }, splitLine: { lineStyle: { color: "#202735" } } },
      yAxis: { type: "category", data: names, axisLabel: { color: "#aab4c3" } },
      series: [
        { type: "bar", name: "Selected", data: utilization, itemStyle: { color: "#d8ff75" } },
        { type: "bar", name: "Router probability", data: probabilities, itemStyle: { color: "#38c6cc" } },
      ],
    };
  }, [events]);
  return <EChart option={option} />;
}

export function TransitionMatrix({ events }: { events: ResearchEvent[] }) {
  const option = useMemo<EChartsOption>(() => {
    const latest = events.filter((event) => event.type === "routing_metrics").at(-1);
    const names = (latest?.expert_names as string[] | undefined) ?? [];
    const matrix = (latest?.transitions as number[][] | undefined) ?? [];
    const points = matrix.flatMap((row, source) => row.map((value, target) => [target, source, value]));
    const max = Math.max(1, ...points.map((point) => Number(point[2])));
    return {
      title: { text: "Expert transition matrix", left: 14, top: 10, textStyle: { color: "#e8edf5", fontSize: 13 } },
      tooltip: { formatter: (params: unknown) => { const value = (params as { value: number[] }).value; return `${names[value[1]]} → ${names[value[0]]}: ${value[2]}`; } },
      grid: { left: 88, right: 30, top: 50, bottom: 58 },
      xAxis: { type: "category", data: names, axisLabel: { color: "#aab4c3", rotate: 25 } },
      yAxis: { type: "category", data: names, axisLabel: { color: "#aab4c3" } },
      visualMap: { min: 0, max, calculable: false, orient: "horizontal", right: 12, top: 4, itemWidth: 10, itemHeight: 70, textStyle: { color: "#77859a" }, inRange: { color: ["#18212c", "#8e69ff", "#f2d276"] } },
      series: [{ type: "heatmap", data: points, label: { show: names.length <= 6, color: "#f4f7fa" } }],
    };
  }, [events]);
  return <EChart option={option} />;
}

export function TrajectoryByStep({ events }: { events: ResearchEvent[] }) {
  const option = useMemo<EChartsOption>(() => {
    const latest = events.filter((event) => event.type === "routing_metrics").at(-1);
    const names = (latest?.expert_names as string[] | undefined) ?? [];
    const rows = (latest?.trajectory_selection_counts as number[][] | undefined) ?? [];
    const totals = rows.map((row) => row.reduce((sum, value) => sum + value, 0));
    return {
      title: { text: "Expert selection by trajectory step", left: 14, top: 10, textStyle: { color: "#e8edf5", fontSize: 13 } },
      tooltip: { trigger: "axis" },
      legend: { top: 10, right: 12, textStyle: { color: "#93a0b2" } },
      grid: { left: 54, right: 22, top: 52, bottom: 38 },
      xAxis: { type: "category", data: rows.map((_, index) => `Step ${index + 1}`), axisLabel: { color: "#aab4c3" } },
      yAxis: { type: "value", max: 1, axisLabel: { color: "#77859a", formatter: (value: number) => `${Math.round(value * 100)}%` }, splitLine: { lineStyle: { color: "#202735" } } },
      series: names.map((name, module) => ({
        name,
        type: "bar",
        stack: "selection",
        data: rows.map((row, step) => (row[module] ?? 0) / Math.max(totals[step] ?? 0, 1)),
      })),
    };
  }, [events]);
  return <EChart option={option} />;
}

export function RefractoryEffect({ events }: { events: ResearchEvent[] }) {
  const option = useMemo<EChartsOption>(() => {
    const latest = events.filter((event) => event.type === "routing_metrics").at(-1);
    const names = (latest?.expert_names as string[] | undefined) ?? [];
    const raw = (latest?.raw_winner_counts as number[] | undefined) ?? [];
    const effective = (latest?.effective_winner_counts as number[] | undefined) ?? [];
    return {
      title: { text: "Refractory effect: raw vs inhibited winner", left: 14, top: 10, textStyle: { color: "#e8edf5", fontSize: 13 } },
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
      legend: { top: 10, right: 12, textStyle: { color: "#93a0b2" } },
      grid: { left: 82, right: 22, top: 50, bottom: 32 },
      xAxis: { type: "value", axisLabel: { color: "#77859a" }, splitLine: { lineStyle: { color: "#202735" } } },
      yAxis: { type: "category", data: names, axisLabel: { color: "#aab4c3" } },
      series: [
        { name: "Raw winner", type: "bar", data: raw, itemStyle: { color: "#38c6cc" } },
        { name: "After inhibition", type: "bar", data: effective, itemStyle: { color: "#d8ff75" } },
      ],
    };
  }, [events]);
  return <EChart option={option} />;
}

export function RoutingRegret({ events }: { events: ResearchEvent[] }) {
  return <RoutingLineChart events={events} title="Counterfactual routing regret" fields={REGRET_FIELDS} />;
}

export function CounterfactualAccuracy({ events }: { events: ResearchEvent[] }) {
  const option = useMemo<EChartsOption>(() => {
    const routing = events.filter((event) => event.type === "routing_metrics");
    return {
      title: { text: "Counterfactual routing accuracy", left: 14, top: 10, textStyle: { color: "#e8edf5", fontSize: 13 } },
      tooltip: { trigger: "axis" },
      legend: { top: 10, right: 12, textStyle: { color: "#93a0b2" } },
      grid: { left: 58, right: 58, top: 52, bottom: 38 },
      xAxis: { type: "value", name: "tokens", axisLabel: { color: "#77859a" }, splitLine: { lineStyle: { color: "#202735" } } },
      yAxis: [{ type: "value", name: "accuracy", max: 1, axisLabel: { color: "#77859a" } }, { type: "value", name: "probes", minInterval: 1, axisLabel: { color: "#77859a" } }],
      series: [
        { name: "Geometric top-1", type: "line", showSymbol: false, connectNulls: true, itemStyle: { color: "#d8ff75" }, data: routing.map((event) => [Number(event.tokens_processed), typeof event.counterfactual_top1_accuracy === "number" ? event.counterfactual_top1_accuracy : null]) },
        { name: "Geometric top-2", type: "line", showSymbol: false, connectNulls: true, itemStyle: { color: "#38c6cc" }, data: routing.map((event) => [Number(event.tokens_processed), typeof event.counterfactual_top2_accuracy === "number" ? event.counterfactual_top2_accuracy : null]) },
        { name: "Probe count", type: "line", yAxisIndex: 1, showSymbol: false, itemStyle: { color: "#f2d276" }, data: routing.map((event) => [Number(event.tokens_processed), typeof event.total_probes === "number" ? event.total_probes : null]) },
      ],
    };
  }, [events]);
  return <EChart option={option} />;
}

export function GeometricMargin({ events }: { events: ResearchEvent[] }) {
  return <RoutingLineChart events={events} title="Geometric action margin" fields={MARGIN_FIELDS} />;
}

function RoutingLineChart({ events, title, fields }: { events: ResearchEvent[]; title: string; fields: RoutingField[] }) {
  const option = useMemo<EChartsOption>(() => {
    const routing = events.filter((event) => event.type === "routing_metrics");
    return {
      title: { text: title, left: 14, top: 10, textStyle: { color: "#e8edf5", fontSize: 13 } },
      tooltip: { trigger: "axis" },
      legend: { top: 10, right: 12, textStyle: { color: "#93a0b2" } },
      grid: { left: 58, right: 22, top: 52, bottom: 38 },
      xAxis: { type: "value", name: "tokens", axisLabel: { color: "#77859a" }, splitLine: { lineStyle: { color: "#202735" } } },
      yAxis: { type: "value", axisLabel: { color: "#77859a" }, splitLine: { lineStyle: { color: "#202735" } } },
      series: fields.map(([name, field, color]) => ({ name, type: "line", showSymbol: false, connectNulls: true, itemStyle: { color }, data: routing.map((event) => [Number(event.tokens_processed), typeof event[field] === "number" ? event[field] : null]) })),
    };
  }, [events, fields, title]);
  return <EChart option={option} />;
}

export function CounterfactualMatrix({ events }: { events: ResearchEvent[] }) {
  const option = useMemo<EChartsOption>(() => {
    const latest = events.filter((event) => event.type === "routing_metrics").at(-1);
    const names = (latest?.expert_names as string[] | undefined) ?? [];
    const matrix = (latest?.counterfactual_matrix as number[][] | undefined) ?? [];
    const points = matrix.flatMap((row, selected) => row.map((value, best) => [best, selected, value]));
    return {
      title: { text: "Selected × counterfactual-best", subtext: "Rows: selected · columns: observed best", left: 14, top: 8, textStyle: { color: "#e8edf5", fontSize: 13 }, subtextStyle: { color: "#77859a", fontSize: 9 } },
      tooltip: { formatter: (params: unknown) => { const value = (params as { value: number[] }).value; return `${names[value[1]]} selected / ${names[value[0]]} best: ${value[2]}`; } },
      grid: { left: 88, right: 26, top: 58, bottom: 54 },
      xAxis: { type: "category", data: names, axisLabel: { color: "#aab4c3", rotate: 25 } },
      yAxis: { type: "category", data: names, axisLabel: { color: "#aab4c3" } },
      visualMap: { min: 0, max: Math.max(1, ...points.map((point) => Number(point[2]))), show: false, inRange: { color: ["#18212c", "#8e69ff", "#f2d276"] } },
      series: [{ type: "heatmap", data: points, label: { show: names.length <= 6, color: "#f4f7fa" } }],
    };
  }, [events]);
  return <EChart option={option} />;
}

export function ExpertWinRate({ events }: { events: ResearchEvent[] }) {
  const option = useMemo<EChartsOption>(() => {
    const latest = events.filter((event) => event.type === "routing_metrics").at(-1);
    const names = (latest?.expert_names as string[] | undefined) ?? [];
    const wins = (latest?.counterfactual_win_rates as number[] | undefined) ?? [];
    const routed = (latest?.actual_routing_rates as number[] | undefined) ?? [];
    const occupied = (latest?.basin_occupancy as number[] | undefined) ?? [];
    return {
      title: { text: "Expert win rate vs basin traffic", left: 14, top: 10, textStyle: { color: "#e8edf5", fontSize: 13 } },
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
      legend: { top: 10, right: 12, textStyle: { color: "#93a0b2" } },
      grid: { left: 82, right: 22, top: 50, bottom: 32 },
      xAxis: { type: "value", max: 1, axisLabel: { color: "#77859a" }, splitLine: { lineStyle: { color: "#202735" } } },
      yAxis: { type: "category", data: names, axisLabel: { color: "#aab4c3" } },
      series: [
        { name: "Counterfactual best", type: "bar", data: wins, itemStyle: { color: "#f2d276" } },
        { name: "Actually routed", type: "bar", data: routed, itemStyle: { color: "#38c6cc" } },
        { name: "Basin occupancy", type: "bar", data: occupied, itemStyle: { color: "#8e69ff" } },
      ],
    };
  }, [events]);
  return <EChart option={option} />;
}

export function GeometryByStep({ events }: { events: ResearchEvent[] }) {
  const option = useMemo<EChartsOption>(() => {
    const latest = events.filter((event) => event.type === "routing_metrics").at(-1);
    const perStep = (latest?.per_step as Record<string, Record<string, number | null>> | undefined) ?? {};
    const steps = Object.keys(perStep).sort((left, right) => Number(left) - Number(right));
    return {
      title: { text: "Counterfactual quality by trajectory step", left: 14, top: 10, textStyle: { color: "#e8edf5", fontSize: 13 } },
      tooltip: { trigger: "axis" },
      legend: { top: 10, right: 12, textStyle: { color: "#93a0b2" } },
      grid: { left: 58, right: 58, top: 52, bottom: 38 },
      xAxis: { type: "category", data: steps.map((step) => `Step ${step}`), axisLabel: { color: "#aab4c3" } },
      yAxis: [{ type: "value", name: "regret", axisLabel: { color: "#77859a" } }, { type: "value", name: "accuracy", max: 1, axisLabel: { color: "#77859a" } }],
      series: [
        { name: "Mean regret", type: "bar", data: steps.map((step) => perStep[step]?.mean_routing_regret), itemStyle: { color: "#ef7b86" } },
        { name: "Top-1 accuracy", type: "line", yAxisIndex: 1, data: steps.map((step) => perStep[step]?.top1_accuracy), itemStyle: { color: "#d8ff75" } },
      ],
    };
  }, [events]);
  return <EChart option={option} />;
}
