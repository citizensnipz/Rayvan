import { useMemo, useState } from "react";
import type { Estimate, ExperimentConfig, ResearchSchema } from "./types";

const formatNumber = (value?: number) => value == null ? "—" : Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 2 }).format(value);

export function ExperimentBuilder({ schema, config, setConfig, estimate, estimating, active, onRun }: {
  schema: ResearchSchema;
  config: ExperimentConfig;
  setConfig: (config: ExperimentConfig) => void;
  estimate?: Estimate;
  estimating: boolean;
  active: boolean;
  onRun: () => void;
}) {
  const [reviewing, setReviewing] = useState(false);
  const suite = schema.suites.find((item) => item.id === config.suite)!;
  const expertCount = Object.values(config.experts).reduce((sum, value) => sum + value, 0);
  const valueRouting = config.architecture === "counterfactual_value_emc";
  const gpuHeavy = !config.routing.value_fit_enabled && Number(config.training.tokens) >= 500_000;
  const usesExperts = ["counterfactual_value_emc", "emc", "sequential_module_aware_emc", "legacy_parallel_emc", "heterogeneous_serial", "old_emc"].includes(config.architecture);
  const setRoot = (field: keyof ExperimentConfig, value: unknown) => setConfig({ ...config, [field]: value });
  const setNested = (group: "routing" | "model" | "training", field: string, value: unknown) => setConfig({ ...config, [group]: { ...config[group], [field]: value } });
  const tags = config.tags.join(", ");
  const routerOnly = valueRouting && config.routing.value_expert_training === "frozen";
  const expertConditioned = valueRouting && config.routing.value_head_type === "expert_geometric";
  const fixedFit = routerOnly && Boolean(config.routing.value_fit_enabled);
  const spectralFit = valueRouting && Boolean(config.routing.value_spectral_comparison);
  const spectralLive = valueRouting && Boolean(config.routing.value_spectral_live);
  const canRun = !active && !estimating && expertCount > 0 && (!routerOnly || Boolean(String(config.routing.value_checkpoint_path ?? "").trim()));
  const architectureLabel = schema.architectures.find((item) => item.id === config.architecture)?.label ?? config.architecture;
  const presetName = useMemo(() => Object.entries(schema.presets).find(([, preset]) => preset.tokens === Number(config.training.tokens))?.[0], [schema, config.training.tokens]);

  const requestRun = () => {
    if (gpuHeavy && !reviewing) { setReviewing(true); return; }
    setReviewing(false);
    onRun();
  };
  const applyModelPreset = (preset: string) => {
    if (preset === "custom") { setNested("model", "preset", preset); return; }
    setConfig({ ...config, model: { ...config.model, ...schema.model_presets[preset], preset } });
  };
  const changeArchitecture = (architecture: string) => {
    const routing = { ...schema.defaults.routing, ...config.routing };
    if (architecture === "counterfactual_value_emc") {
      delete routing.top_k;
      Object.assign(routing, {
        router_type: "counterfactual_value", integrator_type: "identity_free_gate",
        trajectory_steps: 3, routing_geometry_dim: 8, refractory_enabled: false,
        loss_free_balance_enabled: false, balance_coefficient: 0, switch_cost: 0, persistence_bonus: 0,
        value_target: "suffix", value_expert_training: "controlled", value_common_fraction: 0.5,
        value_specialist_temperature: 0.25, value_warmup_steps: 100, value_development_interval: 1,
        value_development_batch_size: 4, value_exploration_rate: 0.1, value_probe_rate: 0.08, value_probe_budget: 1,
        value_router_seed: 0, value_calibration_steps: 64, value_calibration_min_probes: 64,
        value_fixed_reference: true, value_reset_router: true, value_checkpoint_path: "", value_fit_enabled: false, value_head_type: "linear", value_pairwise_weight: 0, value_fit_bank_path: "",
      });
    }
    if (architecture !== "counterfactual_value_emc") { routing.value_checkpoint_path = ""; routing.value_fit_enabled = false; routing.value_fit_bank_path = ""; routing.value_spectral_comparison = false; routing.value_spectral_live = false; }
    if (architecture === "emc" || architecture === "sequential_module_aware_emc") delete routing.top_k;
    if (architecture === "emc") {
      delete routing.top_k;
      routing.router_type = "geometric";
      routing.integrator_type = "acceptance_gate";
      routing.loss_free_balance_enabled = false;
    }
    if (architecture === "sequential_module_aware_emc") { routing.router_type = "module_aware"; routing.integrator_type = "proposal_attention"; }
    if (["legacy_parallel_emc", "old_emc", "heterogeneous_serial"].includes(architecture) && routing.router_type === "counterfactual_value") {
      routing.router_type = "module_aware";
      routing.integrator_type = "proposal_attention";
    }
    if ((architecture === "legacy_parallel_emc" || architecture === "old_emc" || architecture.startsWith("n2_")) && routing.top_k == null) routing.top_k = 2;
    if (architecture === "old_emc" && routing.cycles == null) routing.cycles = 2;
    setConfig({ ...config, architecture, routing,
      training: architecture === "counterfactual_value_emc" && config.training.precision === "fp16"
        ? { ...config.training, precision: "auto" } : config.training,
    });
  };
  const enableFixedFit = (enabled: boolean) => setConfig({ ...config,
    routing: { ...config.routing, value_fit_enabled: enabled, value_fit_bank_path: enabled ? config.routing.value_fit_bank_path ?? "" : "",
      ...(enabled ? { value_expert_training: "frozen", value_fixed_reference: true, value_target: "suffix",
        value_fit_prefixes: 64, value_fit_updates: 1000 } : {}) },
    training: { ...config.training, ...(enabled ? { precision: "fp32", weight_decay: 0, evaluation_interval: 50 } : {}) },
  });
  const changeProbePreset = (preset: string) => setConfig({
    ...config,
    routing: {
      ...config.routing,
      counterfactual_probe_preset: preset,
      counterfactual_probe_fixed_rate: preset === "fixed" ? Number(config.routing.counterfactual_probe_fixed_rate ?? 0.02) : null,
    },
  });

  return <div className="builder-layout">
    <section className="builder-form">
      <div className="section-heading"><span>01</span><div><h2>Experiment identity</h2><p>Saved with the run for exact reproduction.</p></div></div>
      <div className="field-grid two">
        <label><span>Name <em>optional</em></span><input value={config.name} placeholder="e.g. delta routing ablation" onChange={(event) => setRoot("name", event.target.value)} /></label>
        <label><span>Tags <em>comma separated</em></span><input value={tags} placeholder="n1, ablation" onChange={(event) => setRoot("tags", event.target.value.split(",").map((tag) => tag.trim()).filter(Boolean))} /></label>
      </div>
      <label><span>Research notes</span><textarea value={config.notes} rows={2} placeholder="Hypothesis, code change, or expected outcome…" onChange={(event) => setRoot("notes", event.target.value)} /></label>

      <div className="section-heading"><span>02</span><div><h2>Test suite & architecture</h2><p>Backed by existing EMC entrypoints.</p></div></div>
      <div className="choice-grid">
        {schema.suites.map((option) => <button type="button" className={`choice ${config.suite === option.id ? "selected" : ""}`} key={option.id} onClick={() => setRoot("suite", option.id)}><b>{option.label}</b><small>{option.description}</small></button>)}
      </div>
      <div className="task-strip">{suite.tasks?.map((task) => <span key={task}>{task.replaceAll("_", " ")}</span>)}</div>
      <label><span>Architecture</span><select value={config.architecture} onChange={(event) => changeArchitecture(event.target.value)}>{schema.architectures.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}</select></label>

      {usesExperts && <>
        <div className="section-heading"><span>03</span><div><h2>Expert composition</h2><p>Generated from the Python module registry.</p></div></div>
        <div className="expert-grid">{schema.expert_families.map((family) => <label className="expert-control" key={family.id}><span>{family.label}</span><div><button type="button" onClick={() => setRoot("experts", { ...config.experts, [family.id]: Math.max(0, (config.experts[family.id] ?? 0) - 1) })}>−</button><strong>{config.experts[family.id] ?? 0}</strong><button type="button" onClick={() => setRoot("experts", { ...config.experts, [family.id]: (config.experts[family.id] ?? 0) + 1 })}>+</button></div></label>)}</div>
      </>}

      <div className="section-heading"><span>04</span><div><h2>Routing & integration</h2><p>Only controls implemented by the selected backend are shown.</p></div></div>
      {valueRouting && <>
        <label className="toggle"><input type="checkbox" checked={spectralLive} onChange={(e) => setConfig({ ...config,
          routing: { ...config.routing, value_spectral_live: e.target.checked, ...(e.target.checked ? {
            value_spectral_comparison: false, value_expert_training: "frozen", value_fit_enabled: true,
            value_fixed_reference: true, value_reset_router: true, value_target: "suffix", value_head_type: "linear",
            value_pairwise_weight: 0, value_fit_prefixes: 256, value_fit_updates: 100,
            value_spectral_live_updates: 100, value_spectral_online_lr: 0.001, value_spectral_eval_prefixes: 128,
            value_spectral_window: 8, value_spectral_basins: 1, value_router_seed: 17
          } : {}) }, training: { ...config.training, ...(e.target.checked ? {
            precision: "fp32", weight_decay: 0, learning_rate: 0.01, evaluation_interval: 20, batch_size: 4
          } : {}) }
        })} /><span>Spectral geometry — sequential learning + live trajectory charts</span></label>
        {spectralLive && <p>One combined spectral + geometry router controls all trajectory steps, recomputing shape after every expert update. First it learns from your bank, then fresh training prefixes generate counterfactual suffix targets under the current spectral policy. Experts, embeddings, Integrator and readout stay frozen. A fresh fixed evaluation panel compares final loss with your original router and a training-derived constant-per-step policy. No evaluation labels train the router.</p>}
        <label className="toggle"><input type="checkbox" checked={Boolean(config.routing.value_spectral_comparison)} onChange={(e) => setConfig({ ...config,
          routing: { ...config.routing, value_spectral_comparison: e.target.checked, ...(e.target.checked ? {
            value_spectral_live: false,
            value_expert_training: "frozen", value_fit_enabled: true, value_fixed_reference: true, value_reset_router: true,
            value_target: "suffix", value_head_type: "linear", value_pairwise_weight: 0, value_fit_prefixes: 256, value_fit_updates: 100
          } : {}) }, training: { ...config.training, ...(e.target.checked ? { precision: "fp32", weight_decay: 0, learning_rate: 0.01 } : {}) }
        })} /><span>Spectral geometry — frozen-bank comparison (13 variants)</span></label>
        {Boolean(config.routing.value_spectral_comparison) && <p>Creates a bank from your local source checkpoint, or validates and reuses Saved bank path below. Experts and shared layers stay frozen. Windows 8/16/32, spectral-only, geometry-only and combined basins are compared on identical suffix losses. Fitting uses CPU with cached descriptors; the selected device is used for bank measurement. Learning rate and Router seed below apply to all variants. No synthetic data is used. The baseline is legacy mean-pooled geometric routing, not the recent expert-conditioned router.</p>}
        <label><span>Source checkpoint path {routerOnly ? "(required)" : "(optional warm-start)"}</span><input value={String(config.routing.value_checkpoint_path ?? "")} placeholder="Full local path to checkpoints/model-best.pt" onChange={(e) => setNested("routing", "value_checkpoint_path", e.target.value)} /></label>
        <p>From History, open a saved value-EMC run and select “Test router from checkpoint” to copy its model settings. Each router test starts with fresh optimizer state. Keep the model/data seed fixed when changing the router seed.</p>
        {routerOnly && !spectralLive && <p>Experts, Integrator, embeddings and readout stay frozen. With fixed continuation enabled, randomized collection and counterfactual labels are independent of the new router. Final capability-generation diagnostics are skipped; held-out router audits still run.</p>}
      </>}
      {routerOnly && !expertConditioned && !spectralFit && !spectralLive && <label className="toggle"><input type="checkbox" checked={fixedFit} onChange={(e) => enableFixedFit(e.target.checked)} /><span>Fixed-bank fitting diagnostic</span></label>}
      {fixedFit && <>{!spectralLive && <p>Measure both banks once, then repeatedly fit the same training states using full-bank FP32 updates. Experts stay frozen. The normal endpoint budget, batch multiplier, exploration, calibration and probe schedules are unused in this mode. Audit cadence still applies. Keep the source checkpoint fixed; compare need dimensions and prediction heads using the same saved bank.</p>}<div className="field-grid two">
        <NumberField label="Fixed prefixes per split" value={Number(config.routing.value_fit_prefixes ?? 64)} min={1} onChange={(v) => setNested("routing", "value_fit_prefixes", v)} />
        <NumberField label={spectralLive ? "Bank warmup updates" : "Router fitting updates"} value={Number(config.routing.value_fit_updates ?? 1000)} min={1} onChange={(v) => setNested("routing", "value_fit_updates", v)} />
      </div><label><span>Saved bank path (optional)</span><input value={String(config.routing.value_fit_bank_path ?? "")} placeholder="Leave empty to measure; otherwise use checkpoints/router-fit-bank.pt" onChange={(e) => setNested("routing", "value_fit_bank_path", e.target.value)} /></label><p>Reuse requires the same source checkpoint, data seed and prefixes per split. It skips bank measurements.{spectralLive && " Sequential training and trajectory evaluations still execute experts."}</p></>}
      <div className="field-grid three">
        {(config.architecture === "legacy_parallel_emc" || config.architecture === "old_emc" || config.architecture.startsWith("n2_")) && <NumberField label="Top-K" value={Number(config.routing.top_k ?? 2)} min={1} max={Math.max(expertCount, 1)} onChange={(value) => setNested("routing", "top_k", value)} />}
        {config.architecture === "old_emc" && <NumberField label="EMC cycles" value={Number(config.routing.cycles)} min={1} onChange={(value) => setNested("routing", "cycles", value)} />}
        {spectralFit && <>
          <NumberField label="Learned baseline need dimension" value={Number(config.routing.routing_geometry_dim)} min={1} onChange={v => setNested("routing","routing_geometry_dim",v)} />
          <NumberField label="Router seed" value={Number(config.routing.value_router_seed)} min={0} onChange={v => setNested("routing","value_router_seed",v)} />
          <p>Trajectory steps: {String(config.routing.trajectory_steps)} (copied from source). Frozen experts, fixed suffix continuation and router reset are enabled. Each fitting update count applies to each of 13 variants. Spectral initialization is deterministic; another router seed alone may produce identical spectral fits.</p>
        </>}
        {spectralLive && <>
          <NumberField label="Neighbourhood window (tokens)" value={Number(config.routing.value_spectral_window ?? 8)} min={1} onChange={v => setNested("routing","value_spectral_window",v)} />
          <NumberField label="Basins per expert" value={Number(config.routing.value_spectral_basins ?? 1)} min={1} onChange={v => setNested("routing","value_spectral_basins",v)} />
          <NumberField label="Sequential learning updates" value={Number(config.routing.value_spectral_live_updates ?? 100)} min={1} onChange={v => setNested("routing","value_spectral_live_updates",v)} />
          <label className="toggle"><input type="checkbox" checked={Boolean(config.routing.value_spectral_snapshot)} onChange={e => setConfig({...config, routing: {...config.routing, value_spectral_snapshot:e.target.checked, ...(e.target.checked ? {value_spectral_snapshot_states:20,value_spectral_snapshot_updates:20,value_spectral_live_updates:1000} : {})}, training:{...config.training,...(e.target.checked ? {evaluation_interval:100} : {})}})} /><span>Snapshot-batch spectral learning</span></label>
          {Boolean(config.routing.value_spectral_snapshot) && <>
            <NumberField label="Counterfactual states per snapshot" value={Number(config.routing.value_spectral_snapshot_states ?? 20)} min={1} onChange={v => setNested("routing","value_spectral_snapshot_states",v)} />
            <NumberField label="Fit updates per snapshot" value={Number(config.routing.value_spectral_snapshot_updates ?? 20)} min={1} onChange={v => setNested("routing","value_spectral_snapshot_updates",v)} />
            <p>Freeze a copy of the spectral router, collect this many state/depth counterfactuals, then fit the entire batch before refreshing the copy. All later expert choices still use transformed-state spectral geometry. No old-policy labels carry into the next batch. Planned probes: {Math.ceil(Number(config.routing.value_spectral_live_updates)/Number(config.routing.value_spectral_snapshot_updates ?? 20))*Number(config.routing.value_spectral_snapshot_states ?? 20)}. Sequential learning updates count optimizer steps, not probes.</p>
          </>}
          <NumberField label="Sequential learning rate" value={Number(config.routing.value_spectral_online_lr ?? 0.001)} min={0.000001} step={0.0001} onChange={v => setNested("routing","value_spectral_online_lr",v)} />
          <NumberField label="Fresh evaluation prefixes" value={Number(config.routing.value_spectral_eval_prefixes ?? 128)} min={2} onChange={v => setNested("routing","value_spectral_eval_prefixes",v)} />
          <NumberField label="Router seed" value={Number(config.routing.value_router_seed ?? 17)} min={0} onChange={v => setNested("routing","value_router_seed",v)} />
          <p>Trajectory steps: {String(config.routing.trajectory_steps)}, copied from source. KL preference supervision; no cost MSE, balance penalty or refractory inhibition. One uniformly sampled request/depth is probed per online update, comparing all experts. Normal token budget is unused. Validation interval controls learning-chart audits in both phases.</p>
        </>}
        {valueRouting && !spectralFit && !spectralLive && <>
          <label><span>Nexus</span><strong>Centered downstream values</strong></label>
          <label><span>Integrator</span><strong>Identity-free acceptance (starts at 0.5)</strong></label>
          <label><span>Inference</span><strong>Sequential greedy; repeats allowed</strong></label>
          <NumberField label="Trajectory steps" value={Number(config.routing.trajectory_steps)} min={1} onChange={(v) => setNested("routing", "trajectory_steps", v)} />
          <NumberField label="Need dimension" value={Number(config.routing.routing_geometry_dim)} min={1} onChange={(v) => setNested("routing", "routing_geometry_dim", v)} />
          <label><span>Prediction head</span><select value={String(config.routing.value_head_type ?? "linear")} onChange={(e) => setConfig({ ...config, routing: { ...config.routing, value_head_type: e.target.value, ...(e.target.value !== "expert_geometric" ? { value_pairwise_weight: 0 } : {}), ...(["geometric", "expert_geometric"].includes(e.target.value) ? { routing_geometry_dim: 32 } : {}), ...(e.target.value === "expert_geometric" ? { value_expert_training: "frozen", value_fixed_reference: true, value_reset_router: true, value_target: "suffix", value_fit_enabled: false, value_fit_bank_path: "", value_effect_dim: 16, value_effect_weight: 0.01, value_cost_mse_weight: 1, value_replay_capacity: 1024, value_replay_batch_size: 64 } : {}) } })}><option value="linear">Linear</option><option value="mlp">Nonlinear (GELU)</option><option value="geometric">Relational geometric (slots + basins)</option><option value="expert_geometric">Expert-conditioned geometric + effects + replay</option></select></label>
          {config.routing.value_head_type === "geometric" && <>
            <NumberField label="State summary slots" value={Number(config.routing.value_geometry_slots ?? 4)} min={1} onChange={(v) => setNested("routing", "value_geometry_slots", v)} />
            <NumberField label="Competence basins per expert" value={Number(config.routing.value_geometry_prototypes ?? 4)} min={1} onChange={(v) => setNested("routing", "value_geometry_prototypes", v)} />
            <NumberField label="Decision regret weight" value={Number(config.routing.value_geometry_regret_weight ?? 0.01)} min={0} max={1} step={0.01} onChange={(v) => setNested("routing", "value_geometry_regret_weight", v)} />
            <p>Four-head attention reads the observed state into 32-feature slots, compares the slots, then routes toward learned competence basins. Lowest predicted downstream cost wins; routing is recomputed after each expert.</p>
          </>}
          {expertConditioned && <>
            <NumberField label="Competence basins per expert" value={Number(config.routing.value_geometry_prototypes ?? 4)} min={1} onChange={(v) => setNested("routing", "value_geometry_prototypes", v)} />
            <NumberField label="Decision regret weight" value={Number(config.routing.value_geometry_regret_weight ?? 0.01)} min={0} max={1} step={0.01} onChange={(v) => setNested("routing", "value_geometry_regret_weight", v)} />
            <NumberField label="Effect signature dimension" value={Number(config.routing.value_effect_dim ?? 16)} min={1} onChange={(v) => setNested("routing", "value_effect_dim", v)} />
            <NumberField label="Pairwise advantage weight" value={Number(config.routing.value_pairwise_weight ?? 0)} min={0} max={1} step={0.01} onChange={(v) => setNested("routing", "value_pairwise_weight", v)} />
            <NumberField label="Pairwise temperature" value={Number(config.routing.value_pairwise_temperature ?? 0.05)} min={0.001} step={0.01} onChange={(v) => setNested("routing", "value_pairwise_temperature", v)} />
            <NumberField label="Pairwise tie tolerance" value={Number(config.routing.value_pairwise_tie_tolerance ?? 0.001)} min={0} step={0.001} onChange={(v) => setNested("routing", "value_pairwise_tie_tolerance", v)} />
            <NumberField label="Cost MSE weight" value={Number(config.routing.value_cost_mse_weight ?? 1)} min={0} max={1} step={0.01} onChange={(v) => setNested("routing", "value_cost_mse_weight", v)} />
            <NumberField label="Effect prediction weight" value={Number(config.routing.value_effect_weight ?? 0.01)} min={0} max={1} step={0.01} onChange={(v) => setNested("routing", "value_effect_weight", v)} />
            <NumberField label="Replay capacity (states)" value={Number(config.routing.value_replay_capacity ?? 1024)} min={1} onChange={(v) => setNested("routing", "value_replay_capacity", v)} />
            <NumberField label="Replay samples per update" value={Number(config.routing.value_replay_batch_size ?? 64)} min={1} onChange={(v) => setNested("routing", "value_replay_batch_size", v)} />
            <p>Each expert has its own learned attention query and state view. Shared-distance competence basins predict suffix costs; a training-only decoder predicts integrated updates. Fresh probes plus replay train the router. For the pairwise-only test, set Pairwise advantage weight to 1 and Cost MSE, Effect prediction and Decision regret weights to 0. Pairwise supervision compares every measured expert pair, with larger loss gaps receiving more weight. This experiment requires frozen experts and fixed continuation; previous loss-only fitting banks are not used.</p>
          </>}
          {config.routing.value_head_type === "mlp" && <NumberField label="Head hidden dimension" value={Number(config.routing.value_head_hidden_dim ?? 32)} min={1} onChange={(v) => setNested("routing", "value_head_hidden_dim", v)} />}
          <label><span>Counterfactual target</span><select disabled={expertConditioned} value={String(config.routing.value_target ?? "suffix")} onChange={(e) => setNested("routing", "value_target", e.target.value)}><option value="suffix">Final rerouted trajectory (hypothesis)</option><option value="immediate">Immediate result (control)</option></select></label>
          <label><span>Expert learning</span><select disabled={expertConditioned} value={String(config.routing.value_expert_training ?? "controlled")} onChange={(e) => setConfig({ ...config, routing: { ...config.routing, value_expert_training: e.target.value, value_fit_bank_path: e.target.value === "frozen" ? config.routing.value_fit_bank_path ?? "" : "", value_fit_enabled: e.target.value === "frozen" && Boolean(config.routing.value_fit_enabled) } })}><option value="controlled">Controlled common + specialist practice</option><option value="ordinary">Traffic-driven updates (control)</option><option value="frozen">Router only; freeze experts and shared layers</option></select></label>
          <NumberField label="Router seed" value={Number(config.routing.value_router_seed ?? 0)} min={0} onChange={(v) => setNested("routing", "value_router_seed", v)} />
          <NumberField label="Router calibration (blocks)" value={Number(config.routing.value_calibration_steps ?? 64)} min={0} onChange={(v) => setNested("routing", "value_calibration_steps", v)} />
          <NumberField label="Minimum calibration probes" value={Number(config.routing.value_calibration_min_probes ?? 64)} min={0} onChange={(v) => setNested("routing", "value_calibration_min_probes", v)} />
          <label className="toggle"><input type="checkbox" checked={Boolean(config.routing.value_reset_router ?? true)} onChange={(e) => setNested("routing", "value_reset_router", e.target.checked)} /><span>Reset router on checkpoint load</span></label>
          {routerOnly && <label className="toggle"><input type="checkbox" disabled={expertConditioned} checked={Boolean(config.routing.value_fixed_reference ?? true)} onChange={(e) => setNested("routing", "value_fixed_reference", e.target.checked)} /><span>Fixed checkpoint continuation policy</span></label>}
          <NumberField label="Common practice fraction" value={Number(config.routing.value_common_fraction ?? 0.5)} min={0} max={1} step={0.05} onChange={(v) => setNested("routing", "value_common_fraction", v)} />
          <NumberField label="Specialist temperature" value={Number(config.routing.value_specialist_temperature ?? 0.25)} min={0.001} step={0.05} onChange={(v) => setNested("routing", "value_specialist_temperature", v)} />
          <NumberField label="Common-only warmup (blocks)" value={Number(config.routing.value_warmup_steps ?? 100)} min={0} onChange={(v) => setNested("routing", "value_warmup_steps", v)} />
          <NumberField label="Practice interval (blocks)" value={Number(config.routing.value_development_interval ?? 1)} min={1} onChange={(v) => setNested("routing", "value_development_interval", v)} />
          <NumberField label="Practice items per expert" value={Number(config.routing.value_development_batch_size ?? 4)} min={1} onChange={(v) => setNested("routing", "value_development_batch_size", v)} />
          <NumberField label="Training exploration rate" value={Number(config.routing.value_exploration_rate ?? 0.1)} min={0} max={1} step={0.01} onChange={(v) => setNested("routing", "value_exploration_rate", v)} />
          <NumberField label="Probe probability per request" value={Number(config.routing.value_probe_rate ?? 0.08)} min={0} max={1} step={0.01} onChange={(v) => setNested("routing", "value_probe_rate", v)} />
          <NumberField label="Probe cap per block" value={Number(config.routing.value_probe_budget ?? 1)} min={1} onChange={(v) => setNested("routing", "value_probe_budget", v)} />
        </>}
        {config.architecture === "emc" && <label><span>Nexus type</span><strong>Geometric competence basin</strong></label>}
        {config.architecture === "emc" && <label><span>Integrator</span><strong>Learned acceptance gate</strong></label>}
        {config.architecture === "sequential_module_aware_emc" && <label><span>Nexus type</span><strong>Legacy module-aware scorer</strong></label>}
        {["legacy_parallel_emc", "old_emc"].includes(config.architecture) && <label><span>Router</span><select value={String(config.routing.router_type)} onChange={(event) => setNested("routing", "router_type", event.target.value)}><option value="module_aware">Module-aware Nexus</option><option value="fixed_index">Fixed-index baseline</option></select></label>}
        {["legacy_parallel_emc", "old_emc"].includes(config.architecture) && <label><span>Integrator</span><select value={String(config.routing.integrator_type)} onChange={(event) => setNested("routing", "integrator_type", event.target.value)}><option value="proposal_attention">Proposal attention</option><option value="weighted_average">Weighted gate</option></select></label>}
        {["legacy_parallel_emc", "old_emc"].includes(config.architecture) && <NumberField label="Balance loss weight" value={Number(config.routing.balance_coefficient)} step={0.001} min={0} onChange={(value) => setNested("routing", "balance_coefficient", value)} />}
        {["legacy_parallel_emc", "old_emc"].includes(config.architecture) && <NumberField label="Entropy floor" value={Number(config.routing.balance_entropy_floor)} step={0.05} min={0} max={1} onChange={(value) => setNested("routing", "balance_entropy_floor", value)} />}
        {["emc", "sequential_module_aware_emc"].includes(config.architecture) && <>
          <NumberField label="Trajectory steps" value={Number(config.routing.trajectory_steps)} min={1} onChange={(value) => setNested("routing", "trajectory_steps", value)} />
          <label><span>Experts per step</span><strong>1 (fixed)</strong></label>
          <label className="toggle"><input type="checkbox" checked={Boolean(config.routing.refractory_enabled)} onChange={(event) => setNested("routing", "refractory_enabled", event.target.checked)} /><span>Refractory routing</span></label>
          <NumberField label="Inhibition strength" value={Number(config.routing.refractory_strength)} step={0.01} min={0} onChange={(value) => setNested("routing", "refractory_strength", value)} />
          <NumberField label="Inhibition decay" value={Number(config.routing.refractory_decay)} step={0.05} min={0} max={1} onChange={(value) => setNested("routing", "refractory_decay", value)} />
        </>}
        {config.architecture === "sequential_module_aware_emc" && <>
          <NumberField label="Switch cost" value={Number(config.routing.switch_cost)} step={0.01} min={0} onChange={(value) => setNested("routing", "switch_cost", value)} />
          <NumberField label="Persistence bonus" value={Number(config.routing.persistence_bonus)} step={0.01} min={0} onChange={(value) => setNested("routing", "persistence_bonus", value)} />
        </>}
        {config.architecture === "emc" && <>
          <NumberField label="Geometry dimension" value={Number(config.routing.routing_geometry_dim)} min={2} onChange={(value) => setNested("routing", "routing_geometry_dim", value)} />
          <NumberField label="Basins per expert" value={Number(config.routing.competence_prototypes_per_expert)} min={1} onChange={(value) => setNested("routing", "competence_prototypes_per_expert", value)} />
          <label className="toggle"><input type="checkbox" checked={Boolean(config.routing.counterfactual_calibration_enabled)} onChange={(event) => setNested("routing", "counterfactual_calibration_enabled", event.target.checked)} /><span>Counterfactual calibration</span></label>
          <label><span>Probe schedule</span><select value={String(config.routing.counterfactual_probe_preset)} onChange={(event) => changeProbePreset(event.target.value)}><option value="decaying">8% → 2% → 1%</option><option value="fixed">Fixed rate</option></select></label>
          {config.routing.counterfactual_probe_preset === "fixed" && <NumberField label="Fixed probe rate" value={Number(config.routing.counterfactual_probe_fixed_rate ?? 0.02)} step={0.01} min={0} max={1} onChange={(value) => setNested("routing", "counterfactual_probe_fixed_rate", value)} />}
          <label className="toggle"><input type="checkbox" checked={Boolean(config.routing.counterfactual_uncertainty_enabled)} onChange={(event) => setNested("routing", "counterfactual_uncertainty_enabled", event.target.checked)} /><span>Uncertainty-triggered probes</span></label>
          <NumberField label="Uncertainty margin" value={Number(config.routing.counterfactual_uncertainty_margin)} step={0.01} min={0} onChange={(value) => setNested("routing", "counterfactual_uncertainty_margin", value)} />
          <NumberField label="Maximum probe budget" value={Number(config.routing.counterfactual_max_probes_per_forward)} min={0} onChange={(value) => setNested("routing", "counterfactual_max_probes_per_forward", value)} />
          <label className="toggle"><input type="checkbox" checked={Boolean(config.routing.loss_free_balance_enabled)} onChange={(event) => setNested("routing", "loss_free_balance_enabled", event.target.checked)} /><span>Loss-free balancing</span></label>
          <NumberField label="Balance bias LR" value={Number(config.routing.balance_bias_lr)} step={0.001} min={0} onChange={(value) => setNested("routing", "balance_bias_lr", value)} />
          <NumberField label="Balance bias limit" value={Number(config.routing.balance_bias_limit)} step={0.05} min={0} onChange={(value) => setNested("routing", "balance_bias_limit", value)} />
        </>}
      </div>

      <div className="section-heading"><span>05</span><div><h2>Model shape</h2><p>Shared latent and module dimensions.</p></div></div>
      <div className="field-grid three">
        <label><span>Size preset</span><select value={String(config.model.preset)} onChange={(event) => applyModelPreset(event.target.value)}><option value="quick">Quick</option><option value="research">Research</option><option value="custom">Custom</option></select></label>
        <label><span>Comparison matching</span><select value={String(config.model.fairness_mode)} onChange={(event) => setNested("model", "fairness_mode", event.target.value)}><option value="custom">Architecture-specific</option><option value="capacity">Capacity matched</option><option value="compute">Compute matched</option></select></label>
        {usesExperts && <label><span>SSM backend</span><select value={String(config.model.ssm_backend ?? "auto")} onChange={(e) => setNested("model", "ssm_backend", e.target.value)}><option value="auto">Auto: fused CUDA / parallel fallback</option><option value="cuda">Fused CUDA (compiler required)</option><option value="parallel_scan">PyTorch parallel scan</option><option value="reference">Reference recurrence</option></select></label>}
        <NumberField label="Latent width" value={Number(config.model.latent_dim)} min={8} step={8} onChange={(value) => setConfig({ ...config, model: { ...config.model, latent_dim: value, preset: "custom" } })} />
        <NumberField label="Context length" value={Number(config.model.context_length)} min={4} step={4} onChange={(value) => setNested("model", "context_length", value)} />
        <NumberField label="Attention heads" value={Number(config.model.attention_heads)} min={1} onChange={(value) => setNested("model", "attention_heads", value)} />
        <NumberField label="Module hidden width" value={Number(config.model.module_hidden_dim)} min={8} step={8} onChange={(value) => setNested("model", "module_hidden_dim", value)} />
        {["legacy_parallel_emc", "heterogeneous_serial"].includes(config.architecture) && <NumberField label="Chunk size" value={Number(config.model.chunk_size)} min={1} onChange={(value) => setNested("model", "chunk_size", value)} />}
        {config.architecture.startsWith("n2_") && <NumberField label="N1 depth" value={Number(config.model.n1_depth)} min={1} onChange={(value) => setNested("model", "n1_depth", value)} />}
      </div>

      <div className="section-heading"><span>06</span><div><h2>Training</h2><p>Presets populate one editable ExperimentConfig.</p></div></div>
      {valueRouting && !spectralLive && <p>One supervised endpoint per observed prefix. Context, probes and practice add compute and are counted separately. Budget curves use endpoint targets; compare total wall time and expert work across controls. The batch multiplier enlarges the state pool in memory.</p>}
      {spectralLive && <p>Duration is Bank warmup updates + Sequential learning updates. Validation cadence is in router updates; evaluation size is Fresh evaluation prefixes above. Batch size controls fresh training-state collection (safely capped for expert execution).</p>}
      {!spectralLive && <div className="preset-row">{Object.entries(schema.presets).map(([id, preset]) => <button key={id} type="button" className={presetName === id ? "active" : ""} onClick={() => setNested("training", "tokens", preset.tokens)}>{preset.label}</button>)}</div>}
      <div className="field-grid three">
        {!spectralLive && <NumberField label={valueRouting ? "Supervised endpoint budget" : "Training tokens"} value={Number(config.training.tokens)} min={1} step={1000} onChange={(value) => setNested("training", "tokens", value)} />}
        <NumberField label="Batch size" value={Number(config.training.batch_size)} min={1} onChange={(value) => setNested("training", "batch_size", value)} />
        {!spectralLive && <NumberField label={valueRouting ? "State-pool batch multiplier" : "Gradient accumulation"} value={Number(config.training.gradient_accumulation)} min={1} onChange={(value) => setNested("training", "gradient_accumulation", value)} />}
        <NumberField label={spectralLive ? "Bank warmup learning rate" : "Learning rate"} value={Number(config.training.learning_rate)} min={0.000001} step={0.0001} onChange={(value) => setNested("training", "learning_rate", value)} />
        <NumberField label="Weight decay" value={Number(config.training.weight_decay)} min={0} step={0.01} onChange={(value) => setNested("training", "weight_decay", value)} />
        <NumberField label="Model / data seed" value={Number(config.training.seed)} min={0} onChange={(value) => setNested("training", "seed", value)} />
        <label><span>Precision</span><select value={String(config.training.precision)} onChange={(event) => setNested("training", "precision", event.target.value)}><option value="auto">Auto</option><option value="bf16">BF16</option>{!valueRouting && <option value="fp16">FP16</option>}<option value="fp32">FP32</option></select></label>
        <label><span>Device</span><select value={String(config.training.device)} onChange={(event) => setNested("training", "device", event.target.value)}><option value="cuda">CUDA GPU</option><option value="cpu">CPU</option></select></label>
        <NumberField label="Validation cadence (steps)" value={Number(config.training.evaluation_interval)} min={1} onChange={(value) => setNested("training", "evaluation_interval", value)} />
        {!spectralLive && <NumberField label="Validation batches" value={Number(config.training.evaluation_batches ?? 4)} min={1} onChange={(value) => setNested("training", "evaluation_batches", value)} />}
        {!spectralLive && <NumberField label="Telemetry cadence (steps)" value={Number(config.training.telemetry_interval ?? 1)} min={1} onChange={(value) => setNested("training", "telemetry_interval", value)} />}
        {!spectralLive && config.suite === "capability_10" && <NumberField label="Diagnostic examples per capability" value={Number(config.training.diagnostic_examples_per_capability ?? 20)} min={1} onChange={(value) => setNested("training", "diagnostic_examples_per_capability", value)} />}
        {!spectralLive && <label><span>Projection targets <em>{valueRouting ? "endpoints" : "tokens"}, comma separated</em></span><input value={config.projection_targets.join(", ")} onChange={(event) => setRoot("projection_targets", event.target.value.split(",").map((value) => Number(value.trim())).filter((value) => Number.isFinite(value) && value > 0))} /></label>}
      </div>
    </section>

    <aside className="launch-card">
      <p className="eyebrow">Preflight</p><h2>{config.name || "Untitled experiment"}</h2>
      {spectralLive && <p>Total: {Number(config.routing.value_fit_updates)+Number(config.routing.value_spectral_live_updates)} router updates ({String(config.routing.value_fit_updates)} bank + {String(config.routing.value_spectral_live_updates)} sequential).</p>}
      <dl><div><dt>Suite</dt><dd>{suite.label}</dd></div><div><dt>Architecture</dt><dd>{architectureLabel}</dd></div>
        <div><dt>{fixedFit ? "Router updates" : valueRouting ? "Endpoint budget" : "Token budget"}</dt><dd>{(Number(fixedFit ? config.routing.value_fit_updates : config.training.tokens)+(spectralLive ? Number(config.routing.value_spectral_live_updates) : 0)).toLocaleString()}</dd></div>
        <div><dt>Experts</dt><dd>{expertCount}</dd></div><div><dt>{["counterfactual_value_emc", "emc", "sequential_module_aware_emc"].includes(config.architecture) ? "Trajectory" : "Active Top-K"}</dt><dd>{["counterfactual_value_emc", "emc", "sequential_module_aware_emc"].includes(config.architecture) ? `${String(config.routing.trajectory_steps)} sequential steps` : String(config.routing.top_k ?? "—")}</dd></div><div><dt>Total params</dt><dd>{estimating ? "Calculating…" : formatNumber(estimate?.total_parameters)}</dd></div><div><dt>Active params</dt><dd>{formatNumber(estimate?.approximate_active_parameters)}</dd></div><div><dt>{valueRouting ? "FLOPs / context token (proxy)" : "FLOPs / token"}</dt><dd>{formatNumber(estimate?.approximate_flops_per_token)}</dd></div></dl>
      {reviewing && <div className="run-warning"><b>Large-run review</b><p>This will execute {Number(config.training.tokens).toLocaleString()} {valueRouting ? "endpoints" : "tokens"} on {String(config.training.device).toUpperCase()}. Verify the configuration above, then confirm.</p></div>}
      <button className="primary launch" disabled={!canRun} onClick={requestRun}>{active ? "GPU run active" : reviewing ? "Confirm & launch" : "Run experiment"}</button>
      {reviewing && <button className="text-button" onClick={() => setReviewing(false)}>Back to editing</button>}
      <p className="microcopy">Runs execute locally through the shared Python harness. Only one GPU-heavy process can run at once.</p>
    </aside>
  </div>;
}

function NumberField({ label, value, onChange, min, max, step = 1 }: { label: string; value: number; onChange: (value: number) => void; min?: number; max?: number; step?: number }) {
  return <label><span>{label}</span><input type="number" value={value} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} /></label>;
}
