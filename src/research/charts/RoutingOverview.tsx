import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import { EChart } from "./EChart";
import type { ResearchEvent } from "../types";

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
