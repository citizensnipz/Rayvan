import { useMemo } from "react";
import type { EChartsOption, SeriesOption } from "echarts";
import { EChart } from "./EChart";

export interface MetricSeries { name: string; data: Array<[number, number | null]>; dashed?: boolean; color?: string }

export function MetricChart({ title, series, xLabel = "Tokens", yLabel }: { title: string; series: MetricSeries[]; xLabel?: string; yLabel?: string }) {
  const option = useMemo<EChartsOption>(() => ({
    animation: false,
    backgroundColor: "transparent",
    title: { text: title, left: 14, top: 10, textStyle: { color: "#e8edf5", fontSize: 13, fontWeight: 600 } },
    tooltip: { trigger: "axis", backgroundColor: "#151a22", borderColor: "#303949", textStyle: { color: "#f3f6fa" } },
    legend: { top: 10, right: 12, textStyle: { color: "#93a0b2", fontSize: 10 } },
    grid: { left: 52, right: 18, top: 48, bottom: 48 },
    toolbox: { right: 8, bottom: 4, feature: { dataZoom: {}, restore: {} }, iconStyle: { borderColor: "#708099" } },
    dataZoom: [{ type: "inside", filterMode: "none" }],
    xAxis: { type: "value", name: xLabel, nameLocation: "middle", nameGap: 30, axisLabel: { color: "#77859a" }, nameTextStyle: { color: "#77859a" }, splitLine: { lineStyle: { color: "#202735" } } },
    yAxis: { type: "value", name: yLabel, axisLabel: { color: "#77859a" }, nameTextStyle: { color: "#77859a" }, splitLine: { lineStyle: { color: "#202735" } }, scale: true },
    series: series.map((item): SeriesOption => ({
      type: "line",
      name: item.name,
      data: item.data,
      showSymbol: item.data.length < 80,
      symbolSize: 4,
      sampling: "lttb",
      progressive: 1000,
      lineStyle: { width: 2, type: item.dashed ? "dashed" : "solid", color: item.color },
      itemStyle: { color: item.color },
      connectNulls: false,
      markLine: item.dashed && item.data.length ? {
        silent: true,
        symbol: "none",
        label: { formatter: "Measured ends", color: "#9aa6b6", fontSize: 9 },
        lineStyle: { color: "#697688", type: "dotted" },
        data: [{ xAxis: item.data[0][0] }],
      } : undefined,
    })),
  }), [title, series, xLabel, yLabel]);
  return <EChart option={option} />;
}
