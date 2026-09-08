import { MetricChart } from "./charts/MetricChart";
import type { ResearchEvent, RunState } from "./types";

type Row = Record<string, unknown>;
const numeric = (value: unknown) => typeof value === "number" ? value : null;
const format = (value: unknown) => typeof value === "number" ? value.toFixed(5) : "—";

export function SpectralLivePanel({ events, report, state, onCancel, onRouterTest, logs }: {
  events: ResearchEvent[]; report?: Row; state: RunState; onCancel?: () => void; onRouterTest?: () => void; logs: string[];
}) {
  const audits = (report?.history ?? events.filter(e => e.type === "spectral_live_audit")) as Row[];
  const progress = events.filter(e => e.type === "spectral_live_progress").at(-1);
  const last = audits.at(-1);
  const step = Math.max(Number(progress?.step ?? 0), Number(last?.step ?? 0));
  const total = Number(last?.total ?? progress?.total ?? 0);
  const series = (key: string) => audits.map(row => [Number(row.step), numeric(row[key])] as [number,number|null]);
  const counts = (last?.route_counts ?? []) as number[][];
  const names = (report?.expert_names ?? []) as string[];
  const delta = last?.delta_original as Row | undefined;
  return <div className="live-view">
    <section className="panel"><h2>Spectral router — sequential learning</h2>
      <p>{state} · {String(progress?.phase ?? last?.phase ?? "Preparing local checkpoint")}</p>
      {(progress?.snapshot_round != null || Boolean(report?.snapshot_batch_enabled)) && <p>Snapshot round {String(progress?.snapshot_round ?? report?.snapshot_rounds ?? "—")} · counterfactual probes {String(progress?.online_probes ?? report?.online_probes ?? "—")}. {progress?.phase === "collecting snapshot evidence" ? `Collecting ${String(progress.collected_states)} / ${String(progress.collection_states)} states before fitting.` : "Evidence is fixed within each fit batch; the spectral policy is refreshed between batches."}</p>}
      <p>{step} / {total || "—"} router updates. Bank warmup ends at update {String(last?.warmup_updates ?? "—")}; subsequent updates use fresh policy-dependent suffix evidence.</p>
      {onCancel && ["initializing","running","validation"].includes(state) && <button className="danger" onClick={onCancel}>Stop safely</button>}
      {onRouterTest && <button onClick={onRouterTest}>Configure another test from original checkpoint</button>}
      <p>Experts and shared layers are frozen. At every trajectory step the router measures the changed latent and chooses again. Evaluation is greedy and uses a fresh fixed panel excluded from training by exact prefix.</p>
      <p>Final endpoint loss: <b>{format(last?.trajectory_loss)}</b> · Original router: {format(last?.original_loss)} · Constant per step: {format(last?.constant_loss)}</p>
      <p>Paired loss difference vs original: {format(delta?.mean)} (negative favours spectral). Approximate interval: {Array.isArray(delta?.approximate_95_interval) ? delta.approximate_95_interval.map(format).join(" to ") : "—"}.</p>
      <p>Trajectory evaluation speed: {format(last?.evaluation_endpoints_per_second)} endpoint/s · {format(last?.evaluation_context_tokens_per_second)} main-context tok/s. This excludes training/probe time and is not text generation speed.</p>
    </section>
    <section className="chart-grid">
      <MetricChart title="Actual final trajectory loss" xLabel="Router updates" yLabel="Endpoint cross-entropy (nats)" series={[
        {name:"Spectral policy",color:"#38c6cc",data:series("trajectory_loss")},
        {name:"Original router",color:"#8e69ff",data:series("original_loss")},
        {name:"Constant per step",color:"#f2d276",data:series("constant_loss")}]}/>
      <MetricChart title="Fixed-bank routing regret" xLabel="Router updates" yLabel="Extra loss (nats)" series={[
        {name:"Training bank",color:"#d8ff75",data:series("train_regret")},
        {name:"Held-out bank",color:"#38c6cc",data:series("held_regret")},
        {name:"Uniform",color:"#8e69ff",data:series("uniform_regret")},
        {name:"Constant per step",color:"#f2d276",data:series("constant_regret")}]}/>
      <MetricChart title="Fitting versus generalization" xLabel="Router updates" yLabel="Fixed-bank KL divergence" series={[
        {name:"Training bank",color:"#d8ff75",data:series("train_kl")},
        {name:"Held-out bank",color:"#38c6cc",data:series("held_kl")}]}/>
      <MetricChart title="Fresh online training objective" xLabel="Router updates" yLabel="KL + basin regularizer" series={[
        {name:"Online sample",color:"#d8ff75",data:events.filter(e => e.type === "spectral_live_progress" && e.phase === "sequential policy learning").map(e => [Number(e.step),numeric(e.training_objective)])}]}/>
    </section>
    <section className="panel"><h3>Greedy evaluation routes by trajectory step</h3>
      <div className="table-wrap"><table><thead><tr><th>Step</th>{(counts[0] ?? []).map((_,e) => <th key={e}>{names[e] ?? `Expert ${e+1}`}</th>)}</tr></thead>
        <tbody>{counts.map((row,t) => <tr key={t}><td>{t+1}</td>{row.map((n,e) => <td key={e}>{n}</td>)}</tr>)}</tbody></table></div>
      <p>Counts describe choices, not proof of specialization. Bank regret uses the original continuation policy, not the complete new policy. Final trajectory loss executes the spectral policy for all steps. Online suffix labels use a spectral snapshot: refreshed every update in legacy mode, or between batches in snapshot-batch mode. Low KL is not winner accuracy.</p>
      <p>No held-out best-checkpoint selection. Repeated monitoring and correlated text prefixes mean the displayed interval is descriptive. Throughput is not generation tok/s.</p>
      <details><summary>Detailed report and per-prefix outcomes</summary><pre>{JSON.stringify(report ?? last ?? {},null,2)}</pre></details>
      <details><summary>Run logs</summary><pre>{logs.join("\n")}</pre></details>
    </section>
  </div>;
}
