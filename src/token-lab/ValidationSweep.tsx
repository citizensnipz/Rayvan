import {useState} from 'react';
import {EChart} from '../research/charts/EChart';
type Obj=Record<string,any>;
export const sweepDefaults={experiment_mode:'validation_sweep',name:'Specialization Validation Sweep',
 population:'homogeneous',sweep_seed_start:42,sweep_seed_count:3,diagnostic_seed:42,common_pretrain_steps:1000,train_steps:1000,
 batch_size:4,learning_rate:.001,evaluation_samples:2000,calibration_samples_per_task:50,
 latent_dim:64,hidden_dim:128,heads:4,sequence_length:48,window:16,reference_size:128,
 spectral_enabled:true,spectral_rate:1,deep_enabled:false,deep_rate:.05,device:'cpu',
 max_features:5,candidate_count:16,cross_task:true,resume_from:''};
const fmt=(v:any)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(4):'—';

export function SweepSetup({config,setConfig}:{config:Obj;setConfig:(v:any)=>void}){
 const set=(k:string,v:any)=>setConfig((c:Obj)=>({...c,[k]:v}));
 const field=(key:string,label:string,min=1,step=1)=><label key={key}>{label}<input type="number" min={min} step={step} value={config[key]??sweepDefaults[key as keyof typeof sweepDefaults]} onChange={e=>set(key,+e.target.value)}/></label>;
 return <section className="panel" style={{gridColumn:'1 / -1'}}><h3>Specialization Validation Sweep</h3>
  <p>{config.population==='heterogeneous'?'One mixed bank: GPT + SSM + recurrent/GRU + Delta. All four are pretrained on the same general examples through one shared encoder/readout. 0% and 100% task bias; all six cross-family comparisons. Architecture-confounded by design.':'GPT, SSM, recurrent/GRU and Delta modules. Two same-family experts each time; 0% and 100%; no router. These are whole expert modules, not interchangeable FFN activations.'}</p>
  <p><strong>{(config.population==='heterogeneous'?2:8)*config.sweep_seed_count} calibration runs</strong> plus automatic discovery and comparison. One common base per family/seed, reused exactly for its control. Jobs run sequentially on your selected device.</p>
  <div className="lab-fields">
   <label>Population<select aria-label="Sweep population" value={config.population??'homogeneous'} onChange={e=>setConfig((c:Obj)=>({...c,population:e.target.value,resume_from:''}))}><option value="homogeneous">Separate homogeneous family pairs</option><option value="heterogeneous">Mixed GPT + SSM + GRU + Delta</option></select></label>
   <label>Sweep name<input value={config.name} onChange={e=>set('name',e.target.value)}/></label>
   {field('sweep_seed_count','Number of training seeds')}{field('sweep_seed_start','First training seed',0)}{field('diagnostic_seed','Diagnostic seed (fixed across sweep)',0)}
   {field('common_pretrain_steps','Common pretraining steps')}{field('train_steps','Specialist steps per expert')}{field('batch_size','Training batch size')}{field('learning_rate','Learning rate',.000001,.0001)}
   {field('evaluation_samples','Evaluation locations per calibration run',100,10)}{field('calibration_samples_per_task','Separate sanity-check locations per task')}
   {field('latent_dim','Latent dimension',4,4)}{field('hidden_dim','Expert hidden dimension',4,4)}{field('heads','Attention / Delta heads')}{field('sequence_length','Causal prefix length',4)}
   <label>Neighbourhood<select value={config.window} onChange={e=>set('window',+e.target.value)}>{[4,8,16,32].map(n=><option key={n}>{n}</option>)}</select></label>
   {field('reference_size','Training reference-bank size')}{field('max_features','Maximum selected features')}{field('candidate_count','Candidate feature count')}
   <label>Device<select value={config.device} onChange={e=>set('device',e.target.value)}><option>cpu</option><option>cuda</option></select></label>
   <label><input type="checkbox" checked disabled/> Core measurements and raw observations</label>
   <label><input type="checkbox" checked={config.spectral_enabled} onChange={e=>set('spectral_enabled',e.target.checked)}/> Spectral measurements</label>
   {field('spectral_rate','Spectral sampling probability',0,.05)}
   <label><input type="checkbox" checked={config.deep_enabled} onChange={e=>set('deep_enabled',e.target.checked)}/> Deep probes (secondary only)</label>
   {field('deep_rate','Deep sampling probability',0,.05)}
   <label><input type="checkbox" checked={config.cross_task} onChange={e=>set('cross_task',e.target.checked)}/> Leave-one-task-out transfer checks</label>
   <label>Resume previous sweep directory (optional)<input value={config.resume_from??''} onChange={e=>set('resume_from',e.target.value)}/></label>
  </div>
  {config.population==='heterogeneous'&&<p>The 0% control has equal general exposure, not identical experts. At 100%, task partitions rotate across expert slots by seed; these are calibration assignments only, never used in Emergent Specialization. At least four seeds cover a complete rotation. General mixed checkpoints are created automatically; unrelated homogeneous checkpoints cannot be combined.</p>}
  <p>Automatic: balanced 10-task common pretraining; {config.population==='heterogeneous'?'seed-rotated four-way task partition':'calculation vs memory/sequence profile'}; matched budgets; same-stream 0% controls (identity only for homogeneous experts); independent held-out probes; target-free feature discovery; feature-group comparisons; shuffled-label sanity check; offline choice regret; combined report. No checkpoint paths to choose.</p>
  <p>At least 3 seeds recommended (5 for a larger overnight run). Resume requires the same settings; completed child training is reused, analysis is regenerated. One failed family is reported and does not discard the other jobs. No timing guarantee; the first jobs provide a local runtime reference.</p>
 </section>;
}

export function SweepResults({detail,onResume,onRouting,onEmergence}:{detail:Obj;onResume:(config:Obj)=>void;onRouting?:()=>void;onEmergence?:()=>void}){
 const a=detail.analysis.sweep,entries=a.entries??[],[selected,setSelected]=useState(''),[outcome,setOutcome]=useState(''),[selectedPair,setSelectedPair]=useState('');
 const e=entries.find((v:Obj)=>v.key===selected)??entries[0];
 const target=e?.discovery?.[outcome]?outcome:Object.keys(e?.discovery??{})[0],d=e?.discovery?.[target];
 const pairKeys=Object.keys(e?.pair_extras??{}),pair=pairKeys.includes(selectedPair)?selectedPair:pairKeys[0];
 const extra=e?.pair_extras?.[pair]??e?.extra;
 const points=entries.filter((v:Obj)=>v.status==='completed'&&(a.settings?.population==='heterogeneous'||v.strength===1)).map((v:Obj)=>({name:v.key,value:(v.pair_extras?.[pair]??v.extra)?.groups?.all?.decisions?.gain_over_constant??null}));
 return <section className="panel"><h2>Specialization Validation Sweep results</h2>
  <p>{a.status} · {entries.filter((v:Obj)=>v.status==='completed').length}/{a.planned_runs} calibration runs completed. Results are offline diagnostics, not a validated sequential router.</p>
  {onEmergence&&<button onClick={onEmergence}>Test emergence with this general bank</button>}
  {onRouting&&a.settings?.population!=='heterogeneous'&&<button onClick={onRouting}>Test routing with these saved experts</button>}
  <button onClick={()=>onResume({...a.settings,resume_from:detail.runDirectory})}>Prepare resume / reanalysis of this sweep</button>
  {a.settings?.population==='heterogeneous'&&<><p>Mixed architecture baseline: differences at 0% are expected. Mean pair gains are not a four-way routing score. Use Emergent Specialization for actual population routing.</p><details><summary>All six pair comparisons, 0% and 100%, by seed</summary><pre>{JSON.stringify(a.families?.mixed?.pair_replication,null,2)}</pre></details></>}
  <table><thead><tr><th>Family</th><th>Matched controls</th><th>Mean choice gain</th><th>Verdict</th></tr></thead><tbody>{Object.entries(a.families??{}).map(([family,v]:any)=><tr key={family}><td>{family}</td><td>{v.matched_exposure_control_pairs??v.matched_identical_control_pairs}</td><td>{fmt(v.mean_choice_gain)}</td><td>{v.verdict}</td></tr>)}</tbody></table>
  {pairKeys.length>0&&<label>Expert pair for choice gains and feature-group comparisons<select value={pair} onChange={ev=>setSelectedPair(ev.target.value)}>{pairKeys.map(k=><option key={k}>{k}</option>)}</select></label>}
  <EChart replace option={{animation:false,tooltip:{},grid:{left:60,bottom:100},xAxis:{type:'category',data:points.map((v:Obj)=>v.name),axisLabel:{rotate:45}},yAxis:{type:'value',name:'Choice gain (nats)'},series:[{type:'bar',data:points.map((v:Obj)=>v.value)}]}}/>
  <p>Positive gain means a diagnostic predictor’s choices beat the training-selected constant expert on held-out observations. Zero/negative results stay visible. Mixed experts share one common representation; separate homogeneous family runs use distinct bases. Architecture parameter counts differ.</p>
  <label>Inspect family / seed / strength<select value={e?.key??''} onChange={ev=>setSelected(ev.target.value)}>{entries.map((v:Obj)=><option key={v.key} value={v.key}>{v.key} · {v.status}</option>)}</select></label>
  {e&&<><p>{e.error}</p><p>Source: {e.source_directory??'not completed'}</p>
   <p>General base: {e.controls?.common_hash??'unverified'} · shared frozen: {String(e.controls?.shared_unchanged??'unverified')} · exact final identity: {String(e.controls?.exact_control_final_identity??'unverified')}</p>
   {d&&<><label>Prediction outcome<select value={target} onChange={ev=>setOutcome(ev.target.value)}>{Object.keys(e.discovery).map(k=><option key={k}>{k}</option>)}</select></label>
    <p>{d.status} · selected features: {d.selected_features?.join(', ')||'none'}</p>
    <table><thead><tr><th>Predictor</th><th>Test R²</th><th>MAE</th><th>Spearman</th></tr></thead><tbody>{Object.entries(d.scores??{}).map(([k,s]:any)=><tr key={k}><td>{k}</td><td>{fmt(s.r2)}</td><td>{fmt(s.mae)}</td><td>{fmt(s.spearman)}</td></tr>)}</tbody></table>
    <table><thead><tr><th>Removed feature</th><th>MAE increase</th><th>95% interval</th></tr></thead><tbody>{(d.ablations??[]).map((v:Obj)=><tr key={v.feature}><td>{v.feature}</td><td>{fmt(v.mae_increase)}</td><td>{v.mae_increase_ci.map(fmt).join(' … ')}</td></tr>)}</tbody></table>
   </>}
   <h3>Does geometry add beyond simple controls?</h3>
   <table><thead><tr><th>Input set</th><th>Test R²</th><th>MAE</th><th>MAE minus all</th><th>95% interval</th><th>Choice gain</th></tr></thead><tbody>{Object.entries(extra?.groups??{}).map(([k,g]:any)=><tr key={k}><td>{k}</td><td>{fmt(g.scores.r2)}</td><td>{fmt(g.scores.mae)}</td><td>{fmt(g.mae_minus_all)}</td><td>{g.mae_minus_all_ci?.map(fmt).join(' … ')??'—'}</td><td>{fmt(g.decisions.gain_over_constant)}</td></tr>)}</tbody></table>
   <p>Shuffled-label test R²: {fmt(extra?.shuffled_label_check?.scores?.r2)}. A single negative sanity check does not establish statistical significance.</p>
   <h3>Transfer to excluded tasks</h3><table><thead><tr><th>Excluded task</th><th>Test n</th><th>All-input R²</th><th>Controls R²</th><th>All-input choice gain</th></tr></thead><tbody>{(extra?.cross_task??[]).map((t:Obj)=><tr key={t.task}><td>{t.task} {t.status!=='ok'?t.status:''}</td><td>{t.test_n??'—'}</td><td>{fmt(t.all?.scores?.r2)}</td><td>{fmt(t.controls?.scores?.r2)}</td><td>{fmt(t.all?.decisions?.gain_over_constant)}</td></tr>)}</tbody></table>
   <h3>Specialization sanity check</h3><table><thead><tr><th>Expert</th><th>Formed?</th><th>Differential gap</th><th>95% interval</th></tr></thead><tbody>{Object.entries(e.sanity??{}).map(([k,s]:any)=><tr key={k}><td>{k}</td><td>{String(s.specialization_formed)}</td><td>{fmt(s.differential_gap)}</td><td>{s.differential_gap_ci?.map(fmt).join(' … ')}</td></tr>)}</tbody></table>
   <details><summary>Full controls and per-job measurements</summary><pre>{JSON.stringify(e,null,2)}</pre></details></>}
  <h3>Combined evidence report</h3><p>Share report.md and sweep-analysis.json from this parent run. Child runs and raw observations are preserved under jobs/.</p>
  <pre className="lab-report">{detail.report}</pre>
 </section>;
}
