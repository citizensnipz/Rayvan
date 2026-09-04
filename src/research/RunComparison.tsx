import { useMemo, useState } from "react";
import { MetricChart } from "./charts/MetricChart";
import { TaskChart } from "./charts/TaskChart";
import type { ResearchEvent, RunDetail } from "./types";

const colors = ["#d8ff75", "#38c6cc", "#8e69ff", "#f2d276", "#ef7b86"];

export function RunComparison({ runs }: { runs: RunDetail[] }) {
  const [axis, setAxis] = useState<"tokens" | "time">("tokens");
  const seriesFor = (type: string, field: string) => runs.map((run, index) => ({
    name: run.summary?.name ?? run.runId,
    color: colors[index % colors.length],
    data: run.events.filter((event) => event.type === type).map((event) => [axis === "tokens" ? Number(event.tokens_processed) : Number(event.elapsed_seconds), typeof event[field] === "number" ? event[field] as number : null] as [number, number | null]),
  }));
  const mergedTasks = useMemo(() => {
    const result: Record<string, Record<string, unknown>> = {};
    runs.forEach((run) => {
      const tasks = run.diagnostics?.capability_results as Record<string, Record<string, unknown>> | undefined;
      Object.entries(tasks ?? {}).forEach(([task, values]) => { result[`${run.summary?.name ?? run.runId}: ${task}`] = values; });
    });
    return result;
  }, [runs]);
  const scalingPoint = (x: (run: RunDetail) => unknown, y: (run: RunDetail) => unknown) => runs.map((run, index) => ({ name: run.summary?.name ?? run.runId, color: colors[index % colors.length], data: [[Number(x(run)), typeof y(run) === "number" ? y(run) as number : null] as [number, number | null]] }));
  return <section className="compare-view">
    <header className="view-title"><div><p className="eyebrow">Multi-run analysis</p><h1>Run comparison</h1><p>Compatible observations are overlaid; measured data remains separate from projections.</p></div><div className="segmented"><button className={axis === "tokens" ? "active" : ""} onClick={() => setAxis("tokens")}>Tokens</button><button className={axis === "time" ? "active" : ""} onClick={() => setAxis("time")}>Wall time</button></div></header>
    <div className="comparison-table panel"><table><thead><tr><th>Run</th><th>Architecture</th><th>Tokens</th><th>Val loss</th><th>Perplexity</th><th>tok/s</th><th>Total params</th><th>Active params</th><th>FLOPs/token</th><th>Peak VRAM</th></tr></thead><tbody>{runs.map((run) => <tr key={run.runId}><td><b>{run.summary?.name ?? run.runId}</b></td><td>{run.config?.architecture}</td><td>{fmt(run.summary?.headline?.tokens_processed, 0)}</td><td>{fmt(run.summary?.headline?.validation_loss)}</td><td>{fmt(run.summary?.headline?.perplexity, 2)}</td><td>{fmt(run.summary?.headline?.tokens_per_second, 0)}</td><td>{fmt(run.model?.total_parameters, 0)}</td><td>{fmt(run.model?.approximate_active_parameters, 0)}</td><td>{fmt(run.model?.approximate_flops_per_token, 0)}</td><td>{bytes(run.summary?.headline?.peak_vram_bytes)}</td></tr>)}</tbody></table></div>
    <div className="chart-grid"><MetricChart title="Validation loss" xLabel={axis === "tokens" ? "Tokens" : "Seconds"} series={seriesFor("validation", "validation_loss")} /><MetricChart title="Perplexity" xLabel={axis === "tokens" ? "Tokens" : "Seconds"} series={seriesFor("validation", "validation_perplexity")} /><MetricChart title="Throughput" xLabel={axis === "tokens" ? "Tokens" : "Seconds"} series={seriesFor("training_step", "tokens_per_second")} /><MetricChart title="Routing entropy" xLabel={axis === "tokens" ? "Tokens" : "Seconds"} series={seriesFor("routing_metrics", "entropy")} /></div>
    {runs.length >= 2 && <><div className="section-heading"><span>↗</span><div><h2>Historical resource scaling</h2><p>Exploratory relationships across the selected measured runs.</p></div></div><div className="chart-grid"><MetricChart title="Expert count vs throughput" xLabel="Experts" yLabel="tok/s" series={scalingPoint((run) => Object.values(run.config?.experts ?? {}).reduce((sum, value) => sum + value, 0), (run) => run.summary?.headline?.tokens_per_second)} /><MetricChart title="Cycles vs throughput" xLabel="Cycles" yLabel="tok/s" series={scalingPoint((run) => run.config?.routing.cycles, (run) => run.summary?.headline?.tokens_per_second)} /><MetricChart title="Model size vs peak VRAM" xLabel="Total parameters" yLabel="GiB" series={scalingPoint((run) => run.model?.total_parameters, (run) => typeof run.summary?.headline?.peak_vram_bytes === "number" ? run.summary.headline.peak_vram_bytes / 2 ** 30 : null)} /><MetricChart title="Active parameters vs throughput" xLabel="Active parameters" yLabel="tok/s" series={scalingPoint((run) => run.model?.approximate_active_parameters, (run) => run.summary?.headline?.tokens_per_second)} /><MetricChart title="FLOPs/token vs validation loss" xLabel="FLOPs / token" yLabel="Loss" series={scalingPoint((run) => run.model?.approximate_flops_per_token, (run) => run.summary?.headline?.validation_loss)} /><MetricChart title="Wall time vs validation loss" xLabel="Seconds" yLabel="Loss" series={scalingPoint((run) => run.summary?.headline?.runtime_seconds, (run) => run.summary?.headline?.validation_loss)} /></div></>}
    {Object.keys(mergedTasks).length > 0 && <TaskChart tasks={mergedTasks} />}
  </section>;
}

function fmt(value: unknown, digits = 3) { return typeof value === "number" ? Intl.NumberFormat("en", { maximumFractionDigits: digits }).format(value) : "—"; }
function bytes(value: unknown) { return typeof value === "number" ? `${(value / 2 ** 30).toFixed(2)} GiB` : "—"; }
