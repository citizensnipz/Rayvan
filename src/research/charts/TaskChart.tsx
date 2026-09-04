import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import { EChart } from "./EChart";

export function TaskChart({ tasks }: { tasks: Record<string, Record<string, unknown>> }) {
  const option = useMemo<EChartsOption>(() => {
    const entries = Object.entries(tasks);
    return {
      animationDuration: 300,
      title: { text: "10-task diagnostic", left: 14, top: 10, textStyle: { color: "#e8edf5", fontSize: 13 } },
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
      legend: { top: 9, right: 12, textStyle: { color: "#93a0b2" } },
      grid: { left: 138, right: 28, top: 48, bottom: 30 },
      xAxis: { type: "value", min: 0, max: 1, axisLabel: { color: "#77859a" }, splitLine: { lineStyle: { color: "#202735" } } },
      yAxis: { type: "category", data: entries.map(([name]) => name.replaceAll("_", " ")), axisLabel: { color: "#aab4c3" } },
      series: [
        { type: "bar", name: "Exact", data: entries.map(([, value]) => value.exact_accuracy ?? 0), itemStyle: { color: "#d8ff75" } },
        { type: "bar", name: "Token accuracy", data: entries.map(([, value]) => value.token_accuracy ?? 0), itemStyle: { color: "#38c6cc" } },
      ],
    };
  }, [tasks]);
  return <EChart option={option} className="chart chart-tall" />;
}
