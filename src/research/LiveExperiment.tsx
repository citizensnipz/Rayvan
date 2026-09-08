import { useMemo } from "react";
import { SpectralLivePanel } from "./SpectralLivePanel";
import { ExpertHeatmap } from "./charts/ExpertHeatmap";
import { MetricChart, type MetricSeries } from "./charts/MetricChart";
import { CounterfactualAccuracy, CounterfactualMatrix, ExpertWinRate, GeometricMargin, GeometryByStep, RefractoryEffect, RoutingOverview, RoutingRegret, TrajectoryByStep, TransitionMatrix } from "./charts/RoutingOverview";
import { TaskChart } from "./charts/TaskChart";
import type { ResearchEvent, RunDetail, RunState } from "./types";

const number = (value: unknown) => typeof value === "number" ? value : null;
const compact = (value: unknown, digits = 2) => typeof value === "number" ? Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: digits }).format(value) : "—";
const duration = (seconds: unknown) => typeof seconds === "number" ? seconds < 60 ? `${seconds.toFixed(1)}s` : `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s` : "—";

export function LiveExperiment({ events, state, runId, logs, detail, onCancel, onRouterTest }: { events: ResearchEvent[]; state: RunState; runId?: string; logs: string[]; detail?: RunDetail; onCancel?: () => void; onRouterTest?: () => void }) {
  const training = events.filter((event) => event.type === "training_step");
  const validation = events.filter((event) => event.type === "validation");
  const routing = events.filter((event) => event.type === "routing_metrics");
  const latest = training.at(-1) ?? validation.at(-1);
  const latestValidation = validation.at(-1);
  const fixedFit = latest?.objective === "router_fixed_bank" || Boolean(detail?.config?.routing.value_fit_enabled);
  const target = number(latest?.target_tokens) ?? number(fixedFit ? detail?.config?.routing.value_fit_updates : detail?.config?.training.tokens) ?? 0;
  const processed = number(latest?.tokens_processed) ?? number(detail?.summary?.headline?.tokens_processed) ?? 0;
  const progress = target ? Math.min(100, processed / target * 100) : state === "completed" ? 100 : 0;
  const remaining = useMemo(() => {
    const rate = number(latest?.tokens_per_second);
    return rate && target > processed ? (target - processed) / rate : null;
  }, [latest, target, processed]);
  const projectionSeries = projectionLines(detail, events, validation, "fits", "validation_loss", "Bounded loss");
  const perplexityProjectionSeries = projectionLines(detail, events, validation, "perplexity_fits", "validation_perplexity", "PPL = exp(projected loss)", Number.MIN_VALUE);
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
  const valueRouting = (latestRoute?.value_routing ?? detail?.summary?.value_routing) as Record<string, unknown> | undefined;
  const fit = valueRouting?.fixed_bank as Record<string, unknown> | undefined;
  const fitTrain = fit?.train as Record<string, unknown> | undefined;
  const fitHeld = fit?.held_out as Record<string, unknown> | undefined;
  const fitSeries = (split: string, field: string): [number, number | null][] => routing.flatMap((event) => {
    const bank = (event.value_routing as Record<string, unknown> | undefined)?.fixed_bank as Record<string, unknown> | undefined;
    return bank ? [[Number(event.tokens_processed), number((bank[split] as Record<string, unknown> | undefined)?.[field])]] : [];
  });
  const heldOut = valueRouting?.held_out as Record<string, unknown> | undefined;
  const opportunity = (valueRouting?.expert_update_batches ?? []) as number[];
  const practiceItems = (valueRouting?.expert_training_items ?? []) as number[];
  const valueCosts = valueRouting?.costs as Record<string, number> | undefined;
  const valueAudits = routing.filter((event) => {
    const value = event.value_routing as Record<string, unknown> | undefined;
    const audit = value?.held_out as Record<string, unknown> | undefined;
    return audit?.snapshot_version === Number(value?.label_snapshot_version) + 1;
  });
  const isEndpoint = Boolean(valueRouting) || detail?.config?.architecture === "counterfactual_value_emc" || latest?.objective === "prefix_endpoint";
  const unit = fixedFit ? "updates/s" : isEndpoint ? "endpoint/s" : "tok/s";
  const countUnit = fixedFit ? "router updates" : isEndpoint ? "endpoints" : "tokens";
  const contextLength = number(latest?.context_length) ?? number(detail?.config?.model.context_length);
  const contextRate = (event: ResearchEvent) => fixedFit ? null : number(event.context_tokens_per_second) ?? (contextLength == null ? null : Number(event.tokens_per_second) * contextLength);
  const xLabel = fixedFit ? "Router updates" : isEndpoint ? "Supervised endpoints" : "Tokens";
  const geometric = latestRoute?.geometric_routing as Record<string, unknown> | undefined;

  const lossSeries: MetricSeries[] = [
    { name: "Train loss — measured", color: "#d8ff75", data: training.map((event) => [Number(event.tokens_processed), number(event.training_loss)]) },
    { name: "Validation loss — measured", color: "#38c6cc", data: validation.map((event) => [Number(event.tokens_processed), number(event.validation_loss)]) },
    ...projectionSeries,
  ];

  if (detail?.config?.routing.value_spectral_live || events.some(e => e.type === "spectral_live_progress")) {
    return <SpectralLivePanel events={events} report={detail?.summary?.spectral_live} state={state} onCancel={onCancel} onRouterTest={onRouterTest} logs={logs} />;
  }
  if (detail?.config?.routing.value_spectral_comparison || events.some(e => e.type === "spectral_progress")) {
    const event = events.filter(e => e.type === "spectral_progress").at(-1);
    const report = detail?.summary?.spectral_comparison;
    const rows = (report?.results ?? event?.results ?? []) as Array<{variant: string; train: Record<string, number>; held_out: Record<string, number>; dead_basins?: number}>;
    const bank = (report?.bank ?? event?.bank ?? {}) as Record<string, unknown>;
    return <div className="live-view"><section className="panel"><h2>Spectral geometry — frozen-bank comparison</h2>
      <p>{state} · {String(event?.phase ?? "Preparing local checkpoint and bank")}</p>
      <p>{String(event?.completed ?? 0)} / {String(event?.total ?? "—")} router updates · {rows.length}/13 variants complete</p>
      <p>Elapsed fitting: {duration(event?.elapsed_seconds)} · {typeof event?.elapsed_seconds === "number" && event.elapsed_seconds > 0 ? compact(Number(event.completed)/event.elapsed_seconds) : "—"} updates/s (includes descriptor setup and variant audits)</p>
      {onCancel && ["initializing","running","validation"].includes(state) && <button onClick={onCancel}>Stop safely</button>}
      <p>Fixed-reference suffix regret, not language loss or perplexity. No experts execute during fitting. Cached fitting runs on CPU; tok/s is not applicable.</p>
      <p>Bank: {String(bank.bank_file ?? "pending")} · {bank.bank_reused ? "reused" : "measured"}</p>
      <p>Fingerprint: {String(bank.bank_sha256 ?? "pending")}</p>
      <p>Compare held-out regret with both uniform and the training-derived constant-per-step baseline. This does not yet test deployment on trajectories chosen by the new router.</p>
      <div className="table-wrap"><table><thead><tr><th>Variant</th><th>Train regret</th><th>Held-out regret ↓</th><th>Uniform</th><th>Constant/step</th><th>Top-1</th><th>Pairwise</th><th>Unused basins</th></tr></thead><tbody>
      {rows.map(row => <tr key={row.variant}><td>{row.variant}</td><td>{compact(row.train.routing_regret,5)}</td><td>{compact(row.held_out.routing_regret,5)}</td><td>{compact(row.held_out.uniform_random_regret,5)}</td><td>{compact(row.held_out.constant_per_step_regret,5)}</td><td>{formatPercent(row.held_out.oracle_top1)}</td><td>{formatPercent(row.held_out.pairwise_ranking_accuracy)}</td><td>{row.dead_basins ?? "—"}</td></tr>)}
      </tbody></table></div><p>Full report, per-step counts, geometry diagnostics and timings are saved in spectral-report.json in this run's directory.</p>
      <details><summary>Detailed results: per-step routing, basins and descriptor diagnostics</summary><pre>{JSON.stringify(report ?? event?.results ?? [], null, 2)}</pre></details>
      <details><summary>Run log</summary><pre>{logs.join("\n")}</pre></details>
      </section></div>;
  }
  return <div className="live-view">
    <section className="run-header panel">
      <div><p className="eyebrow">{detail ? "Stored run" : "Live experiment"}</p><h2>{detail?.summary?.name ?? runId ?? "Waiting for a run"}</h2><div className="state-line"><span className={`state-dot ${state}`} />{state}<small>{runId}</small></div></div>
      {onRouterTest && <button className="primary" onClick={onRouterTest}>Test router from checkpoint</button>}
      {onCancel && ["initializing", "running", "validation", "diagnostics"].includes(state) && <button className="danger" onClick={onCancel}>Stop safely</button>}
      <div className="progress-wrap"><div><span>{processed.toLocaleString()} / {target.toLocaleString()} {countUnit}</span><span>{progress.toFixed(1)}%</span></div><div className="progress"><i style={{ width: `${progress}%` }} /></div></div>
    </section>

    <section className="metric-cards">
      <Metric label={fixedFit ? "Training-bank MSE" : "Train loss"} value={compact(latest?.training_loss, fixedFit ? 7 : 4)} />
      <Metric label={fixedFit ? "Held-out bank MSE" : "Validation loss"} value={compact(latestValidation?.validation_loss ?? detail?.summary?.headline?.validation_loss, fixedFit ? 7 : 4)} accent />
      {!fixedFit && <Metric label="Perplexity" value={compact(latestValidation?.validation_perplexity ?? detail?.summary?.headline?.perplexity, 3)} />}
      <Metric label={fixedFit ? "Router fitting throughput" : isEndpoint ? "Supervised endpoints" : "Training throughput"} value={`${compact(latest?.tokens_per_second ?? detail?.summary?.headline?.tokens_per_second, 2)} ${unit}`} />
      {isEndpoint && !fixedFit && <Metric label="Main context throughput" value={`${compact(latest ? contextRate(latest) : detail?.summary?.headline?.context_tokens_per_second, 2)} tok/s`} />}
      {isEndpoint && <Metric label="Router phase" value={String(valueRouting?.router_phase ?? "initializing")} />}
      {isEndpoint && <Metric label="Training probes" value={compact(valueRouting?.total_training_probes, 0)} />}
      {isEndpoint && <Metric label="Initial routing regret" value={compact((valueRouting?.initial_held_out as Record<string, unknown> | undefined)?.mean_regret, 4)} />}
      {isEndpoint && <Metric label="Constant-per-step regret" value={compact(heldOut?.constant_regret, 4)} />}
      {isEndpoint && <Metric label="Uniform routing regret" value={compact(heldOut?.uniform_regret, 4)} />}
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

    {isEndpoint && !fixedFit && <p>Main context tok/s = endpoint/s × context length. This counts primary input-token exposures, excludes repeated expert/probe work, and is not generation tok/s. Both rates include training-loop overhead and periodic validation; initial setup/compilation is excluded.</p>}
    {fixedFit && <section className="panel"><h3>Can the router fit a fixed set of measured answers?</h3>
      <p>Normalized MSE of 1 means no improvement over predicting equal competence. Training-bank error tests fitting; held-out error tests generalization. These are fixed-policy measurements, not full-trajectory language performance.</p>
      <div className="metric-cards"><Metric label="Train normalized MSE" value={compact(fitTrain?.normalized_mse, 4)} /><Metric label="Held-out normalized MSE" value={compact(fitHeld?.normalized_mse, 4)} /><Metric label="Encoder gradient" value={compact(fit?.encoder_gradient_norm, 7)} /><Metric label="Value-head gradient" value={compact(fit?.head_gradient_norm, 7)} /><Metric label="Router parameter change" value={compact(fit?.router_parameter_change_norm, 7)} /></div>
      <p>{String(fit?.prefixes_per_split ?? "—")} unique prefixes / {String(fit?.states_per_split ?? "—")} states per bank. Expert measurement setup: {duration(fit?.setup_seconds)}. No experts execute during fitting.</p>
      {typeof fit?.bank_file === "string" && <label><span>Saved bank path</span><input readOnly value={fit.bank_file} onFocus={(e) => e.currentTarget.select()} /></label>}
      <p>Bank fingerprint: <code>{String(fit?.bank_sha256 ?? "—")}</code>. {fit?.bank_reused ? "Loaded existing bank." : "Measured new bank."}</p>
    </section>}
    <section className="chart-grid">
      {fixedFit && <MetricChart xLabel={xLabel} title="Fitting versus generalization" yLabel="MSE / equal-expert MSE" series={[{name:"Training bank",color:"#d8ff75",data:fitSeries("train","normalized_mse")},{name:"Held-out bank",color:"#38c6cc",data:fitSeries("held_out","normalized_mse")}]} />}
      {fixedFit && <MetricChart xLabel={xLabel} title="Fixed-bank decision regret" yLabel="nats" series={[{name:"Training bank",color:"#d8ff75",data:fitSeries("train","mean_regret")},{name:"Held-out bank",color:"#38c6cc",data:fitSeries("held_out","mean_regret")}]} />}
      <MetricChart xLabel={xLabel} title={fixedFit ? "Fixed-bank MSE (not language loss)" : "Loss & scaling projection"} series={lossSeries} yLabel="Loss" />
      {!fixedFit && <MetricChart xLabel={xLabel} title="Perplexity · projected from loss" series={[{ name: "Validation PPL — measured", color: "#8e69ff", data: validation.map((event) => [Number(event.tokens_processed), number(event.validation_perplexity)]) }, ...perplexityProjectionSeries]} yLabel="PPL" />}
      <MetricChart xLabel={xLabel} title="Training throughput" series={[{ name: fixedFit ? "Router updates / second" : isEndpoint ? "Endpoints / second" : "Tokens / second", color: "#38c6cc", data: training.map((event) => [Number(event.tokens_processed), number(event.tokens_per_second)]) }]} yLabel={unit} />
      {isEndpoint && !fixedFit && <MetricChart xLabel={xLabel} title="Main context throughput" series={[{ name: "Context tok/s", color: "#8e69ff", data: training.map((event) => [Number(event.tokens_processed), contextRate(event)]) }]} yLabel="tok/s" />}
      <MetricChart xLabel={xLabel} title="Step performance" series={[{ name: "Step duration", color: "#f2d276", data: training.map((event) => [Number(event.tokens_processed), number(event.step_time_seconds)]) }]} yLabel="seconds" />
      <MetricChart xLabel={xLabel} title="Learning rate & gradient norm" series={[{ name: "Learning rate", color: "#8e69ff", data: training.map((event) => [Number(event.tokens_processed), number(event.learning_rate)]) }, { name: "Gradient norm", color: "#ef7b86", data: training.map((event) => [Number(event.tokens_processed), number(event.gradient_norm)]) }]} />
      <MetricChart xLabel={xLabel} title="GPU utilization & VRAM" series={[{ name: "GPU %", color: "#d8ff75", data: training.map((event) => [Number(event.tokens_processed), number((event.system as Record<string, unknown> | undefined)?.gpu_utilization_percent)]) }, { name: "VRAM GiB", color: "#38c6cc", data: training.map((event) => [Number(event.tokens_processed), bytesToGiB(event.gpu_memory_used_bytes)]) }]} />
      <MetricChart xLabel={xLabel} title="Routing entropy & integrator" series={[{ name: "Entropy", color: "#d8ff75", data: routing.map((event) => [Number(event.tokens_processed), number(event.entropy)]) }, { name: "Gate magnitude", color: "#8e69ff", data: routing.map((event) => [Number(event.tokens_processed), number(event.mean_gate_magnitude)]) }, { name: "Latent update", color: "#38c6cc", data: routing.map((event) => [Number(event.tokens_processed), number(event.mean_update_norm)]) }]} />
      {valueRouting && <MetricChart xLabel={xLabel} title="Held-out suffix routing quality" series={[
        { name: "Decision regret (nats)", color: "#38c6cc", data: valueAudits.map((event) => [Number(event.tokens_processed), number(((event.value_routing as Record<string, unknown>).held_out as Record<string, unknown>).mean_regret)]) },
        { name: "Cost-gap RMSE", color: "#d8ff75", data: valueAudits.map((event) => [Number(event.tokens_processed), number(((event.value_routing as Record<string, unknown>).held_out as Record<string, unknown>).gap_rmse)]) },
      ]} />}
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

    {projectionFits.length > 0 && <section className="panel projection-quality"><div><p className="eyebrow">Exploratory projection · not measured truth</p><h3>Fit quality and runtime estimates</h3></div><div className="table-wrap"><table><thead><tr><th>Target</th><th>Model</th><th>Predicted loss</th><th>R²</th><th>Points</th><th>Confidence</th><th>Warning</th></tr></thead><tbody>{projectionFits.map((fit, index) => <tr key={index}><td>{Number(fit.prediction_target).toLocaleString()} {countUnit}</td><td>{String(fit.model_type).replaceAll("_", " ")}</td><td>{Number(fit.predicted_value).toFixed(4)}</td><td>{Number(fit.r_squared).toFixed(3)}</td><td>{String(fit.measured_points)}</td><td><span className={`confidence ${fit.confidence}`}>{String(fit.confidence)}</span></td><td>{fit.warning ? String(fit.warning) : "—"}</td></tr>)}</tbody></table></div>{runtimeEstimates.length > 0 && <div className="runtime-projections">{runtimeEstimates.map((item, index) => <span key={index}><b>{Number(item.target_tokens).toLocaleString()} {countUnit}</b>{duration(item.estimated_total_seconds)} total · {String(item.confidence)}</span>)}</div>}</section>}

    {valueRouting && <section className="panel">
      <h3>{fixedFit ? "Fixed-bank router fitting diagnostic" : "Counterfactual value experiment · prefix endpoint objective"}</h3>
      <p>Snapshot {String(valueRouting.label_snapshot_version ?? "—")} · held-out suffix probes {String(heldOut?.probe_count ?? "—")}. Audit target: {String(heldOut?.target ?? "pending")}. Validation labels never train the router.</p>
      <div className="table-wrap"><table><thead><tr><th>Expert</th><th>Training route share</th><th>Update batches</th><th>Practice / routed items</th></tr></thead><tbody>{opportunity.map((updates, i) => <tr key={i}><td>{expertNames[i] ?? `Expert ${i + 1}`}</td><td>{formatPercent(utilization[i])}</td><td>{updates}</td><td>{practiceItems[i]}</td></tr>)}</tbody></table></div>
      {heldOut && <div className="table-wrap"><table><thead><tr><th>Audit step</th><th>Greedy expert counts</th><th>Regret</th><th>Constant expert baseline</th></tr></thead><tbody>{((heldOut.by_depth ?? []) as Array<Record<string, unknown>>).map((row) => <tr key={Number(row.depth)}><td>{Number(row.depth) + 1}</td><td>{((row.route_counts ?? []) as number[]).map((count, index) => `${expertNames[index] ?? index}: ${count}`).join(" · ")}</td><td>{compact(row.regret, 4)}</td><td>{row.constant_expert == null ? "—" : expertNames[Number(row.constant_expert)]}</td></tr>)}</tbody></table></div>}
      <p>Training route shares reflect exploration. Greedy audit counts above show learned choices. The constant-per-step baseline is chosen from training probes only.</p>
      <p>SSM execution: {JSON.stringify(valueRouting.ssm_backends ?? {})}. Collection exploration: {formatPercent(valueRouting.collection_exploration)}.</p>
      <p>Traffic concentration alone does not establish collapse. Check held-out regret, capability quality and controlled update counts.</p>
      <dl>{Object.entries(valueCosts ?? {}).map(([key, value]) => <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{value.toLocaleString()}</dd></div>)}</dl>
    </section>}
    {!valueRouting && expertNames.length > 0 && <section className="panel expert-diagnostic"><div><p className="eyebrow">Starvation diagnostic</p><h3>{starvation.length ? `${starvation.length} expert${starvation.length > 1 ? "s" : ""} below 1%` : "No expert starvation detected"}</h3></div><div className="expert-pills">{expertNames.map((name, index) => <span className={utilization[index] < 0.01 ? "starved" : ""} key={name}>{name} {(utilization[index] * 100).toFixed(1)}%</span>)}</div></section>}
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

function projectionLines(detail: RunDetail | undefined, events: ResearchEvent[], validation: ResearchEvent[], fitField: string, measuredField: string, label: string, lowerBound = 0): MetricSeries[] {
  const predictionEvents = events.filter((event) => event.type === "projection_update");
  const stored = detail?.projections?.predictions ?? [];
  const latest = (predictionEvents.at(-1) ?? stored.at(-1)) as Record<string, unknown> | undefined;
  const fits = (latest?.[fitField] as Array<Record<string, unknown>> | undefined) ?? [];
  if (!fits.length || !validation.length) return [];
  const end = validation.at(-1)!;
  const endPoint: [number, number | null] = [Number(end.tokens_processed), number(end[measuredField])];
  const projected = fits
    .map((fit) => [Number(fit.prediction_target), Math.max(lowerBound, Number(fit.predicted_value))] as [number, number])
    .filter(([target, value]) => Number.isFinite(target) && Number.isFinite(value))
    .sort((left, right) => left[0] - right[0]);
  if (!projected.length) return [];
  const fit = fits[0];
  return [{
    name: `${label} — projected (${String(fit.model_type).replaceAll("_", " ")}, ${String(fit.confidence)} confidence)`,
    dashed: true,
    color: "#f2d276",
    data: [endPoint, ...projected],
  }];
}
