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
  const gpuHeavy = Number(config.training.tokens) >= 500_000;
  const usesExperts = ["emc", "legacy_parallel_emc", "heterogeneous_serial", "old_emc"].includes(config.architecture);
  const setRoot = (field: keyof ExperimentConfig, value: unknown) => setConfig({ ...config, [field]: value });
  const setNested = (group: "routing" | "model" | "training", field: string, value: unknown) => setConfig({ ...config, [group]: { ...config[group], [field]: value } });
  const tags = config.tags.join(", ");
  const canRun = !active && !estimating && expertCount > 0;
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
    const routing = { ...config.routing };
    if (architecture === "emc") delete routing.top_k;
    if ((architecture === "legacy_parallel_emc" || architecture === "old_emc" || architecture.startsWith("n2_")) && routing.top_k == null) routing.top_k = 2;
    if (architecture === "old_emc" && routing.cycles == null) routing.cycles = 2;
    setConfig({ ...config, architecture, routing });
  };

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
      <div className="field-grid three">
        {(config.architecture === "legacy_parallel_emc" || config.architecture === "old_emc" || config.architecture.startsWith("n2_")) && <NumberField label="Top-K" value={Number(config.routing.top_k ?? 2)} min={1} max={Math.max(expertCount, 1)} onChange={(value) => setNested("routing", "top_k", value)} />}
        {config.architecture === "old_emc" && <NumberField label="EMC cycles" value={Number(config.routing.cycles)} min={1} onChange={(value) => setNested("routing", "cycles", value)} />}
        {["emc", "legacy_parallel_emc", "old_emc"].includes(config.architecture) && <label><span>Router</span><select value={String(config.routing.router_type)} onChange={(event) => setNested("routing", "router_type", event.target.value)}><option value="module_aware">Module-aware Nexus</option><option value="fixed_index">Fixed-index baseline</option></select></label>}
        {["emc", "legacy_parallel_emc", "old_emc"].includes(config.architecture) && <label><span>Integrator</span><select value={String(config.routing.integrator_type)} onChange={(event) => setNested("routing", "integrator_type", event.target.value)}><option value="proposal_attention">Proposal gate</option><option value="weighted_average">Weighted gate</option></select></label>}
        {["legacy_parallel_emc", "old_emc"].includes(config.architecture) && <NumberField label="Balance loss weight" value={Number(config.routing.balance_coefficient)} step={0.001} min={0} onChange={(value) => setNested("routing", "balance_coefficient", value)} />}
        {["legacy_parallel_emc", "old_emc"].includes(config.architecture) && <NumberField label="Entropy floor" value={Number(config.routing.balance_entropy_floor)} step={0.05} min={0} max={1} onChange={(value) => setNested("routing", "balance_entropy_floor", value)} />}
        {config.architecture === "emc" && <>
          <NumberField label="Trajectory steps" value={Number(config.routing.trajectory_steps)} min={1} onChange={(value) => setNested("routing", "trajectory_steps", value)} />
          <label><span>Experts per step</span><strong>1 (fixed)</strong></label>
          <NumberField label="Switch cost" value={Number(config.routing.switch_cost)} step={0.01} min={0} onChange={(value) => setNested("routing", "switch_cost", value)} />
          <NumberField label="Persistence bonus" value={Number(config.routing.persistence_bonus)} step={0.01} min={0} onChange={(value) => setNested("routing", "persistence_bonus", value)} />
          <label className="toggle"><input type="checkbox" checked={Boolean(config.routing.refractory_enabled)} onChange={(event) => setNested("routing", "refractory_enabled", event.target.checked)} /><span>Refractory routing</span></label>
          <NumberField label="Inhibition strength" value={Number(config.routing.refractory_strength)} step={0.01} min={0} onChange={(value) => setNested("routing", "refractory_strength", value)} />
          <NumberField label="Inhibition decay" value={Number(config.routing.refractory_decay)} step={0.05} min={0} max={1} onChange={(value) => setNested("routing", "refractory_decay", value)} />
          <label className="toggle"><input type="checkbox" checked={Boolean(config.routing.loss_free_balance_enabled)} onChange={(event) => setNested("routing", "loss_free_balance_enabled", event.target.checked)} /><span>Loss-free balancing</span></label>
          <NumberField label="Balance bias LR" value={Number(config.routing.balance_bias_lr)} step={0.001} min={0} onChange={(value) => setNested("routing", "balance_bias_lr", value)} />
          <NumberField label="Balance bias limit" value={Number(config.routing.balance_bias_limit)} step={0.05} min={0} onChange={(value) => setNested("routing", "balance_bias_limit", value)} />
        </>}
      </div>

      <div className="section-heading"><span>05</span><div><h2>Model shape</h2><p>Shared latent and module dimensions.</p></div></div>
      <div className="field-grid three">
        <label><span>Size preset</span><select value={String(config.model.preset)} onChange={(event) => applyModelPreset(event.target.value)}><option value="quick">Quick</option><option value="research">Research</option><option value="custom">Custom</option></select></label>
        <label><span>Comparison matching</span><select value={String(config.model.fairness_mode)} onChange={(event) => setNested("model", "fairness_mode", event.target.value)}><option value="custom">Architecture-specific</option><option value="capacity">Capacity matched</option><option value="compute">Compute matched</option></select></label>
        <NumberField label="Latent width" value={Number(config.model.latent_dim)} min={8} step={8} onChange={(value) => setConfig({ ...config, model: { ...config.model, latent_dim: value, preset: "custom" } })} />
        <NumberField label="Context length" value={Number(config.model.context_length)} min={4} step={4} onChange={(value) => setNested("model", "context_length", value)} />
        <NumberField label="Attention heads" value={Number(config.model.attention_heads)} min={1} onChange={(value) => setNested("model", "attention_heads", value)} />
        <NumberField label="Module hidden width" value={Number(config.model.module_hidden_dim)} min={8} step={8} onChange={(value) => setNested("model", "module_hidden_dim", value)} />
        {["legacy_parallel_emc", "heterogeneous_serial"].includes(config.architecture) && <NumberField label="Chunk size" value={Number(config.model.chunk_size)} min={1} onChange={(value) => setNested("model", "chunk_size", value)} />}
        {config.architecture.startsWith("n2_") && <NumberField label="N1 depth" value={Number(config.model.n1_depth)} min={1} onChange={(value) => setNested("model", "n1_depth", value)} />}
      </div>

      <div className="section-heading"><span>06</span><div><h2>Training</h2><p>Presets populate one editable ExperimentConfig.</p></div></div>
      <div className="preset-row">{Object.entries(schema.presets).map(([id, preset]) => <button key={id} type="button" className={presetName === id ? "active" : ""} onClick={() => setNested("training", "tokens", preset.tokens)}>{preset.label}</button>)}</div>
      <div className="field-grid three">
        <NumberField label="Training tokens" value={Number(config.training.tokens)} min={1} step={1000} onChange={(value) => setNested("training", "tokens", value)} />
        <NumberField label="Batch size" value={Number(config.training.batch_size)} min={1} onChange={(value) => setNested("training", "batch_size", value)} />
        <NumberField label="Gradient accumulation" value={Number(config.training.gradient_accumulation)} min={1} onChange={(value) => setNested("training", "gradient_accumulation", value)} />
        <NumberField label="Learning rate" value={Number(config.training.learning_rate)} min={0.000001} step={0.0001} onChange={(value) => setNested("training", "learning_rate", value)} />
        <NumberField label="Weight decay" value={Number(config.training.weight_decay)} min={0} step={0.01} onChange={(value) => setNested("training", "weight_decay", value)} />
        <NumberField label="Seed" value={Number(config.training.seed)} min={0} onChange={(value) => setNested("training", "seed", value)} />
        <label><span>Precision</span><select value={String(config.training.precision)} onChange={(event) => setNested("training", "precision", event.target.value)}><option value="auto">Auto</option><option value="bf16">BF16</option><option value="fp16">FP16</option><option value="fp32">FP32</option></select></label>
        <label><span>Device</span><select value={String(config.training.device)} onChange={(event) => setNested("training", "device", event.target.value)}><option value="cuda">CUDA GPU</option><option value="cpu">CPU</option></select></label>
        <NumberField label="Validation cadence (steps)" value={Number(config.training.evaluation_interval)} min={1} onChange={(value) => setNested("training", "evaluation_interval", value)} />
        <label><span>Projection targets <em>tokens, comma separated</em></span><input value={config.projection_targets.join(", ")} onChange={(event) => setRoot("projection_targets", event.target.value.split(",").map((value) => Number(value.trim())).filter((value) => Number.isFinite(value) && value > 0))} /></label>
      </div>
    </section>

    <aside className="launch-card">
      <p className="eyebrow">Preflight</p><h2>{config.name || "Untitled experiment"}</h2>
      <dl><div><dt>Suite</dt><dd>{suite.label}</dd></div><div><dt>Architecture</dt><dd>{architectureLabel}</dd></div><div><dt>Token budget</dt><dd>{Number(config.training.tokens).toLocaleString()}</dd></div><div><dt>Experts</dt><dd>{expertCount}</dd></div><div><dt>{config.architecture === "emc" ? "Trajectory" : "Active Top-K"}</dt><dd>{config.architecture === "emc" ? `${String(config.routing.trajectory_steps)} sequential steps` : String(config.routing.top_k ?? "—")}</dd></div><div><dt>Total params</dt><dd>{estimating ? "Calculating…" : formatNumber(estimate?.total_parameters)}</dd></div><div><dt>Active params</dt><dd>{formatNumber(estimate?.approximate_active_parameters)}</dd></div><div><dt>FLOPs / token</dt><dd>{formatNumber(estimate?.approximate_flops_per_token)}</dd></div></dl>
      {reviewing && <div className="run-warning"><b>Large-run review</b><p>This will execute {Number(config.training.tokens).toLocaleString()} tokens on {String(config.training.device).toUpperCase()}. Verify the configuration above, then confirm.</p></div>}
      <button className="primary launch" disabled={!canRun} onClick={requestRun}>{active ? "GPU run active" : reviewing ? "Confirm & launch" : "Run experiment"}</button>
      {reviewing && <button className="text-button" onClick={() => setReviewing(false)}>Back to editing</button>}
      <p className="microcopy">Runs execute locally through the shared Python harness. Only one GPU-heavy process can run at once.</p>
    </aside>
  </div>;
}

function NumberField({ label, value, onChange, min, max, step = 1 }: { label: string; value: number; onChange: (value: number) => void; min?: number; max?: number; step?: number }) {
  return <label><span>{label}</span><input type="number" value={value} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} /></label>;
}
