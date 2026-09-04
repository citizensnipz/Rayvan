import * as echarts from "echarts/core";
import { BarChart, HeatmapChart, LineChart } from "echarts/charts";
import { DataZoomComponent, GridComponent, LegendComponent, MarkLineComponent, TitleComponent, ToolboxComponent, TooltipComponent, VisualMapComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import { useEffect, useRef } from "react";
import type { EChartsCoreOption } from "echarts/core";

echarts.use([LineChart, BarChart, HeatmapChart, DataZoomComponent, GridComponent, LegendComponent, MarkLineComponent, TitleComponent, ToolboxComponent, TooltipComponent, VisualMapComponent, CanvasRenderer]);

export function EChart({ option, className = "chart" }: { option: EChartsCoreOption; className?: string }) {
  const host = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | undefined>(undefined);

  useEffect(() => {
    if (!host.current) return;
    chart.current = echarts.init(host.current, undefined, { renderer: "canvas" });
    const observer = new ResizeObserver(() => chart.current?.resize());
    observer.observe(host.current);
    return () => {
      observer.disconnect();
      chart.current?.dispose();
    };
  }, []);

  useEffect(() => {
    chart.current?.setOption(option, { notMerge: false, lazyUpdate: true });
  }, [option]);

  return <div ref={host} className={className} />;
}
