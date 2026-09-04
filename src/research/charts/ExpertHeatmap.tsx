import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import { EChart } from "./EChart";
import type { ResearchEvent } from "../types";

export function ExpertHeatmap({ events }: { events: ResearchEvent[] }) {
  const option = useMemo<EChartsOption>(() => {
    const routing = events.filter((event) => event.type === "routing_metrics");
    const names = (routing.at(-1)?.expert_names as string[] | undefined) ?? [];
    const points: number[][] = [];
    routing.forEach((event, x) => ((event.utilization as number[] | undefined) ?? []).forEach((value, y) => points.push([x, y, value])));
    return {
      animation: false,
      title: { text: "Expert utilization heatmap", left: 14, top: 10, textStyle: { color: "#e8edf5", fontSize: 13 } },
      tooltip: { position: "top", formatter: (params: unknown) => { const value = (params as { value: number[] }).value; return `${names[value[1]] ?? `Expert ${value[1]}`}<br/>${(value[2] * 100).toFixed(1)}%`; } },
      grid: { left: 92, right: 28, top: 46, bottom: 42 },
      xAxis: { type: "category", data: routing.map((event) => Number(event.tokens_processed).toLocaleString()), axisLabel: { color: "#77859a", hideOverlap: true }, splitArea: { show: true } },
      yAxis: { type: "category", data: names, axisLabel: { color: "#aab4c3" }, splitArea: { show: true } },
      visualMap: { min: 0, max: 1, calculable: true, orient: "horizontal", right: 12, top: 4, itemWidth: 10, itemHeight: 80, textStyle: { color: "#77859a" }, inRange: { color: ["#18212c", "#1ca2a7", "#d8ff75"] } },
      series: [{ type: "heatmap", data: points, progressive: 1000, emphasis: { itemStyle: { shadowBlur: 8, shadowColor: "#000" } } }],
    };
  }, [events]);
  return <EChart option={option} />;
}
