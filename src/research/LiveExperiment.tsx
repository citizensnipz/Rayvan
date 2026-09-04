import { useMemo } from "react";
import { ExpertHeatmap } from "./charts/ExpertHeatmap";
import { MetricChart, type MetricSeries } from "./charts/MetricChart";
import { CounterfactualAccuracy, CounterfactualMatrix, ExpertWinRate, GeometricMargin, GeometryByStep, RefractoryEffect, RoutingOverview, RoutingRegret, TrajectoryByStep, TransitionMatrix } from "./charts/RoutingOverview";
import { TaskChart } from "./charts/TaskChart";
import type { ResearchEvent, RunDetail, RunState } from "./types";

const number = (value: unknown) => typeof value === "number" ? value : null;
const compact = (value: unknown, digits = 2) => typeof value === "number" ? Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: digits }).format(value) : "—";
const duration = (seconds: unknown) => typeof seconds === "number" ? seconds < 60 ? `${seconds.toFixed(1)}s` : `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s` : "—";

export function LiveExperiment({ events, state, runId, logs, detail, onCancel }: { events: ResearchEvent[]; state: RunState; runId?: string; logs: string[]; detail?: RunDetail; onCancel?: () => void }) {
  const training = events.filter((event) => event.type === "training_step");
  const validation = events.filter((event) => event.type === "validation");
  const routing = events.filter((event) => event.type === "routing_metrics");
  const latest = training.at(-1) ?? validation.at(-1);
  const latestValidation = validation.at(-1);
  const target = number(latest?.target_tokens) ?? number(detail?.config?.training.tokens) ?? 0;
  const processed = number(latest?.tokens_processed) ?? number(detail?.summary?.headline?.tokens_processed) ?? 0;
  const progress = target ? Math.min(100, processed / target * 100) : state === "completed" ? 100 : 0;
  const remaining = useMemo(() => {
    const rate = number(latest?.tokens_per_second);
    return rate && target > processed ? (target - processed) / rate : null;
  }, [latest, target, processed]);
  const projectionSeries = projectionLines(detail, events, validation, "fits", "validation_loss");
  const perplexityProjectionSeries = projectionLines(detail, events, validation, "perplexity_fits", "validation_perplexity");
  const diagnosticEvent = [...events].reverse().find((event: ResearchEvent) => event.type === "diagnostic_result" && event.tasks);
  const tasks = ((diagnosticEvent?.tasks ?? detail?.diagnostics?.capability_results) || {}) as Record<string, Record<string, unknown>>;
  const warnings = events.flatMap((event) => Array.isArray(event.warnings) ? event.warnings as Array<Record<string, string>> : []);
  const latestRoute = routing.at(-1);
  const projectionRecord = (events.filter((event) => event.type === "projection_update").at(-1) ?? detail?.projections?.predictions?.at(-1)) as Record<string, unknown> | undefined;
  const projectionFits = (projectionRecord?.fits as Array<Record<string, unknown>> | undefined) ?? [];
  const runtimeEstimates = (projectionRecord?.runtime_estimates as Array<Record<string, unknown>> | undefined) ?? [];
  const utilization = (latestRoute?.utilization as number[] | undefined) ?? [];
  const expertNames = (latestRoute?.expert_names as string[] | undefined) ?? [];
  const starvation = utilization.map((value, index) => ({ value, name: expertNames[index] ?? `m${index}` })).filter((row) => row.value < 0.01);
  const geometric = latestRoute?.geometric_routing as Record<string, unknown> | undefined;

  const lossSeries: MetricSeries[] = [
    { name: "Train (measured)", color: "#d8ff75", data: training.map((event) => [Number(event.tokens_processed), number(event.training_loss)]) },
    { name: "Validation (measured)", color: "#38c6cc", data: validation.map((event) => [Number(event.tokens_processed), number(event.validation_loss)]) },
    ...projectionSeries,
  ];

  return <div className="live-view">
    <section className="run-header panel">
      <div><p className="eyebrow">{detail ? "Stored run" : "Live experiment"}</p><h2>{detail?.summary?.name ?? runId ?? "Waiting for a run"}</h2><div className="state-line"><span className={`state-dot ${state}`} />{state}<small>{runId}</small></div></div>
      {onCancel && ["initializing", "running", "validation", "diagnostics"].includes(state) && <button className="danger" onClick={onCancel}>Stop safely</button>}
      <div className="progress-wrap"><div><span>{processed.toLocaleString()} / {target.toLocaleString()} tokens</span><span>{progress.toFixed(1)}%</span></div><div className="progress"><i style={{ width: `${progress}%` }} /></div></div>
    </section>

    <section className="metric-cards">
      <Metric label="Train loss" value={compact(latest?.training_loss, 4)} />
      <Metric label="Validation loss" value={compact(latestValidation?.validation_loss ?? detail?.summary?.headline?.validation_loss, 4)} accent />
      <Metric label="Perplexity" value={compact(latestValidation?.validation_perplexity ?? detail?.summary?.headline?.perplexity, 3)} />
      <Metric label="Throughput" value={`${compact(latest?.tokens_per_second ?? detail?.summary?.headline?.tokens_per_second, 2)} tok/s`} />
      <Metric label="Elapsed" value={duration(latest?.elapsed_seconds ?? detail?.summary?.headline?.runtime_seconds)} />
      <Metric label="ETA" value={duration(remaining)} />
      <Metric label="GPU" value={latest?.system && number((latest.system as Record<string, unknown>).gpu_utilization_percent) != null ? `${number((latest.system as Record<string, unknown>).gpu_utilization_percent)}%` : "Unavailable"} />
      <Metric label="VRAM" value={formatBytes(latest?.gpu_memory_used_bytes ?? (latest?.system && (latest.system as Record<string, unknown>).vram_used_bytes))} />
      <Metric label="Same-expert continuation" value={formatPercent(latestRoute?.same_expert_continuation_rate)} />
      <Metric label="Winner changed by inhibition" value={formatPercent(latestRoute?.refractory_changed_winner_rate)} />
      {geometric && <Metric label="Routing regret" value={compact(latestRoute?.mean_routing_regret, 4)} />}
      {geometric && <Metric label="Counterfactual top-1" value={formatPercent(latestRoute?.counterfactual_top1_accuracy)} />}
      {geometric && <Metric label="Counterfactual probes" value={compact(latestRoute?.total_probes, 0)} />}
    </section>

    {warnings.length > 0 && <section className="warning-list panel"><p className="eyebrow">Diagnostic warnings</p>{warnings.slice(-6).map((warning, index) => <div key={`${warning.code}-${index}`}><b>{warning.code?.replaceAll("_", " ")}</b><span>{warning.message ?? String(warning)}</span></div>)}</section>}

    <section className="chart-grid">
      <MetricChart title="Loss & scaling projection" series={lossSeries} yLabel="Loss" />
      <MetricChart title="Perplexity" series={[{ name: "Validation (measured)", color: "#8e69ff", data: validation.map((event) => [Number(event.tokens_processed), number(event.validation_perplexity)]) }, ...perplexityProjectionSeries]} yLabel="PPL" />
      <MetricChart title="Training throughput" series={[{ name: "Tokens / second", color: "#38c6cc", data: training.map((event) => [Number(event.tokens_processed), number(event.tokens_per_second)]) }]} yLabel="tok/s" />
      <MetricChart title="Step performance" series={[{ name: "Step duration", color: "#f2d276", data: training.map((event) => [Number(event.tokens_processed), number(event.step_time_seconds)]) }]} yLabel="seconds" />
      <MetricChart title="Learning rate & gradient norm" series={[{ name: "Learning rate", color: "#8e69ff", data: training.map((event) => [Number(event.tokens_processed), number(event.learning_rate)]) }, { name: "Gradient norm", color: "#ef7b86", data: training.map((event) => [Number(event.tokens_processed), number(event.gradient_norm)]) }]} />
      <MetricChart title="GPU utilization & VRAM" series={[{ name: "GPU %", color: "#d8ff75", data: training.map((event) => [Number(event.tokens_processed), number((event.system as Record<string, unknown> | undefined)?.gpu_utilization_percent)]) }, { name: "VRAM GiB", color: "#38c6cc", data: training.map((event) => [Number(event.tokens_processed), bytesToGiB(event.gpu_memory_used_bytes)]) }]} />
      <MetricChart title="Routing entropy & integrator" series={[{ name: "Entropy", color: "#d8ff75", data: routing.map((event) => [Number(event.tokens_processed), number(event.entropy)]) }, { name: "Gate magnitude", color: "#8e69ff", data: routing.map((event) => [Number(event.tokens_processed), number(event.mean_gate_magnitude)]) }, { name: "Latent update", color: "#38c6cc", data: routing.map((event) => [Number(event.tokens_processed), number(event.mean_update_norm)]) }]} />
      <RoutingOverview events={events} />
      <TrajectoryByStep events={events} />
      <ExpertHeatmap events={events} />
      <TransitionMatrix events={events} />
      <RefractoryEffect events={events} />
      {geometric && <RoutingRegret events={events} />}
      {geometric && <CounterfactualAccuracy events={events} />}
      {geometric && <GeometricMargin events={events} />}
      {geometric && <CounterfactualMatrix events={events} />}
      {geometric && <ExpertWinRate events={events} />}
      {geometric && <GeometryByStep events={events} />}
    </section>

    {projectionFits.length > 0 && <section className="panel projection-quality"><div><p className="eyebrow">Exploratory projection · not measured truth</p><h3>Fit quality and runtime estimates</h3></div><div className="table-wrap"><table><thead><tr><th>Target</th><th>Model</th><th>Predicted loss</th><th>R²</th><th>Points</th><th>Confidence</th><th>Warning</th></tr></thead><tbody>{projectionFits.map((fit, index) => <tr key={index}><td>{Number(fit.prediction_target).toLocaleString()} tokens</td><td>{String(fit.model_type).replaceAll("_", " ")}</td><td>{Number(fit.predicted_value).toFixed(4)}</td><td>{Number(fit.r_squared).toFixed(3)}</td><td>{String(fit.measured_points)}</td><td><span className={`confidence ${fit.confidence}`}>{String(fit.confidence)}</span></td><td>{fit.warning ? String(fit.warning) : "—"}</td></tr>)}</tbody></table></div>{runtimeEstimates.length > 0 && <div className="runtime-projections">{runtimeEstimates.map((item, index) => <span key={index}><b>{Number(item.target_tokens).toLocaleString()} tokens</b>{duration(item.estimated_total_seconds)} total · {String(item.confidence)}</span>)}</div>}</section>}

    {expertNames.length > 0 && <section className="panel expert-diagnostic"><div><p className="eyebrow">Starvation diagnostic</p><h3>{starvation.length ? `${starvation.length} expert${starvation.length > 1 ? "s" : ""} below 1%` : "No expert starvation detected"}</h3></div><div className="expert-pills">{expertNames.map((name, index) => <span className={utilization[index] < 0.01 ? "starved" : ""} key={name}>{name} {(utilization[index] * 100).toFixed(1)}%</span>)}</div></section>}
    {Object.keys(tasks).length > 0 && <><TaskChart tasks={tasks} /><section className="table-wrap task-table"><table><thead><tr><th>Task</th><th>Loss</th><th>Exact</th><th>Token score</th><th>Perplexity</th><th>Samples</th><th>Tokens</th><th>Elapsed</th></tr></thead><tbody>{Object.entries(tasks).map(([task, values]) => <tr key={task}><td>{task.replaceAll("_", " ")}</td><td>{formatMetric(values.cross_entropy)}</td><td>{formatPercent(values.exact_accuracy)}</td><td>{formatPercent(values.token_accuracy)}</td><td>{formatMetric(values.perplexity)}</td><td>{String(values.examples ?? "—")}</td><td>{typeof values.evaluated_tokens === "number" ? values.evaluated_tokens.toLocaleString() : "—"}</td><td>{duration(values.elapsed_seconds)}</td></tr>)}</tbody></table></section></>}

    <details className="logs panel"><summary>Process logs & raw errors <span>{logs.length || (detail?.logs ? detail.logs.split("\n").length : 0)} lines</span></summary><pre>{[...logs, detail?.logs ?? ""].filter(Boolean).join("\n") || "No process output."}</pre></details>
    {detail && <details className="logs panel"><summary>Saved configuration & metadata</summary><pre>{JSON.stringify({ run_directory: detail.runDirectory, config: detail.config, metadata: detail.metadata, model: detail.model, projections: detail.projections }, null, 2)}</pre></details>}
  </div>;
}

function Metric({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) { return <div className={`metric ${accent ? "accent" : ""}`}><span>{label}</span><strong>{value}</strong></div>; }
function bytesToGiB(value: unknown) { return typeof value === "number" ? value / 2 ** 30 : null; }
function formatBytes(value: unknown) { const gib = bytesToGiB(value); return gib == null ? "—" : `${gib.toFixed(2)} GiB`; }
function formatMetric(value: unknown) { return typeof value === "number" ? value.toFixed(4) : "—"; }
function formatPercent(value: unknown) { return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—"; }

function projectionLines(detail: RunDetail | undefined, events: ResearchEvent[], validation: ResearchEvent[], fitField: string, measuredField: string): MetricSeries[] {
  const predictionEvents = events.filter((event) => event.type === "projection_update");
  const stored = detail?.projections?.predictions ?? [];
  const latest = (predictionEvents.at(-1) ?? stored.at(-1)) as Record<string, unknown> | undefined;
  const fits = (latest?.[fitField] as Array<Record<string, unknown>> | undefined) ?? [];
  if (!fits.length || !validation.length) return [];
  const end = validation.at(-1)!;
  const endPoint: [number, number | null] = [Number(end.tokens_processed), number(end[measuredField])];
  return fits.slice(0, 4).map((fit) => ({
    name: `Projected ${String(fit.model_type)} (${String(fit.confidence)})`,
    dashed: true,
    color: "#f2d276",
    data: [endPoint, [Number(fit.prediction_target), number(fit.predicted_value)]],
  }));
}
