import {useState} from 'react';
import {EChart} from '../research/charts/EChart';
type Obj=Record<string,any>;
const fmt=(v:any)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(4):'—';

export function CalibrationSetup({config,setConfig,tasks}:{config:Obj;setConfig:(v:any)=>void;tasks:string[]}){
 const set=(k:string,v:any)=>setConfig((c:Obj)=>({...c,[k]:v}));
 const custom=config.specialization_profile==='custom';
 const preset=(name:string)=>{
  const groups=[['arithmetic','symbolic','program_execution'],['associative_recall','fuzzy_recall','selective_copying','working_memory','compression','stateful_action']];
  const weights=Object.fromEntries(config.families.map((_:string,i:number)=>[String(i),Object.fromEntries(tasks.map(t=>[t,name==='two_way'?Number((groups[i]??[]).includes(t)):Number(tasks[i]===t)]))]));
  setConfig((c:Obj)=>({...c,specialization_profile:name,task_weights:weights}));
 };
 const weight=(i:number,t:string)=>custom?(config.task_weights?.[i]?.[t]??0):config.specialization_profile==='one_task'?Number(tasks[i]===t):Number((i===0?['arithmetic','symbolic','program_execution']:['associative_recall','fuzzy_recall','selective_copying','working_memory','compression','stateful_action']).includes(t));
 return <section><h3>Forced Specialization Calibration</h3><p>Controlled task exposure, not emergent specialization. Common pretraining → identical clones → expert-only fine-tuning → freeze → blind analysis → label reveal. Capability suite and independent topology are required.</p>
  <div className="lab-fields">
   <label>Common base source<select value={config.common_base_source??'pretrain'} onChange={e=>set('common_base_source',e.target.value)}><option value="pretrain">Balanced common pretraining</option><option value="checkpoint">Saved calibration common-base.pt</option></select></label>
   {config.common_base_source==='checkpoint'&&<label>Common checkpoint path<input value={config.common_checkpoint??''} onChange={e=>set('common_checkpoint',e.target.value)}/></label>}
   <label>Common pretraining steps<input type="number" min={1} value={config.common_pretrain_steps??1000} onChange={e=>set('common_pretrain_steps',+e.target.value)}/></label>
   <label>Specialization strength<input type="number" min={0} max={1} step={.05} value={config.specialization_strength??1} onChange={e=>set('specialization_strength',+e.target.value)}/></label>
   <label>Strength preset<select value={config.specialization_strength??1} onChange={e=>set('specialization_strength',+e.target.value)}>{[0,.25,.5,.75,1].map(s=><option value={s} key={s}>{s*100}%{s===0?' · negative control':''}</option>)}</select></label>
   <label>Specialist profile<select value={config.specialization_profile??'two_way'} onChange={e=>preset(e.target.value)}><option value="two_way">Two-way: calculation vs memory/sequence</option><option value="one_task">One task per expert</option><option value="custom">Custom weights</option></select></label>
   <label>Sanity-check examples per task<input type="number" min={1} value={config.calibration_samples_per_task??50} onChange={e=>set('calibration_samples_per_task',+e.target.value)}/></label>
   <label>Success top fraction<input type="number" min={.01} max={.49} step={.05} value={config.success_quantile??.2} onChange={e=>set('success_quantile',+e.target.value)}/></label>
   <label><input type="checkbox" checked={config.identical_stream??true} onChange={e=>set('identical_stream',e.target.checked)}/> Identical minibatch stream at strength 0</label>
   <label>Rerun saved specialists (run directory; optional)<input value={config.specialist_checkpoint??''} onChange={e=>set('specialist_checkpoint',e.target.value)}/></label>
  </div>
  <h4>Task distribution · experimental partitions, not cognitive roles</h4><div className="lab-table"><table><thead><tr><th>Task</th>{config.families.map((_:string,i:number)=><th key={i}>Expert {i+1}</th>)}</tr></thead><tbody>{tasks.map(t=><tr key={t}><td>{t}</td>{config.families.map((_:string,i:number)=><td key={i}><input aria-label={`${t} expert ${i+1} weight`} type="number" min={0} step={.1} disabled={!custom} value={weight(i,t)} onChange={e=>set('task_weights',{...config.task_weights,[i]:{...config.task_weights?.[i],[t]:+e.target.value}})}/></td>)}</tr>)}</tbody></table></div>
  <p>P(task) = (1 − strength) / {tasks.length} + strength × normalized task weight. Equal optimizer steps, endpoint targets and fixed-length context tokens per specialist. Short examples are rejected within the selected task. Only expert parameters update; shared machinery is verified unchanged.</p>
  <p>Calibration uses answer-token endpoints, not arbitrary prompt characters. No mixed-family calibration. AdamW, zero weight decay, gradient clip 1, float32, constant learning rate. Reuse specialists by reopening their run, then entering its directory above; only analysis settings may change.</p>
 </section>;
}

export function CalibrationResults({detail,runs}:{detail:Obj;runs:Obj[]}){
 const [revealed,setRevealed]=useState(false),[expert,setExpert]=useState(''),[feature,setFeature]=useState('effective_rank');
 const a=detail.analysis.calibration,ids=Object.keys(a.experts),id=expert||ids[0],d=a.experts[id],p=d.probes;
 const comparable=runs.filter(r=>r.calibration?.comparison_key===detail.summary?.calibration?.comparison_key).sort((a,b)=>a.calibration.strength-b.calibration.strength);
 const scatter=(detail.rows??[]).filter((r:Obj)=>r.expert_id===id&&typeof r[feature]==='number').map((r:Obj)=>[r[feature],r.improvement]);
 const current=[...new Set([feature,...d.held_out_associations.map((v:Obj)=>v.feature)])];
 const tasks=Object.keys(a.reveal[id].task_matrix);
 const matrixData=ids.flatMap((e,i)=>tasks.map((t,j)=>[i,j,a.reveal[e].task_matrix[t].mean_improvement]));
 const extent=Math.max(.001,...matrixData.map(v=>Math.abs(v[2])));
 const matrix={animation:false,tooltip:{},grid:{left:150,bottom:60},xAxis:{type:'category',data:ids},yAxis:{type:'category',data:tasks},visualMap:{min:-extent,max:extent,calculable:true,orient:'horizontal',bottom:0},series:[{type:'heatmap',data:matrixData}]};
 return <section className="panel"><h2>Calibration · blind signals</h2><p>{a.caveat}</p><label>Specialist<select value={id} onChange={e=>setExpert(e.target.value)}>{ids.map(e=><option key={e}>{e}</option>)}</select></label>
  <label>Pre-expert attribute<select value={feature} onChange={e=>setFeature(e.target.value)}>{a.blind_features.map((k:string)=><option key={k}>{k}</option>)}</select></label>
  <EChart replace option={{animation:false,tooltip:{},xAxis:{type:'value',name:feature,scale:true},yAxis:{type:'value',name:'Improvement',scale:true},series:[{type:'scatter',large:true,symbolSize:4,itemStyle:{opacity:.3},data:scatter}]}}/>
  <p>Scatter shows the saved display subset, including analysis fitting observations. The table below uses test observations only; candidates selected on analysis-train.</p>
  <table><thead><tr><th>Feature</th><th>Test n</th><th>Test Spearman</th><th>95% interval</th></tr></thead><tbody>{d.held_out_associations.map((v:Obj)=><tr key={v.feature}><td>{v.feature}</td><td>{v.n}</td><td>{fmt(v.spearman)}</td><td>{v.interval?.map(fmt).join(' … ')??'—'}</td></tr>)}</tbody></table>
  <h3>Detectability · held-out probes</h3><table><thead><tr><th>Inputs / model</th><th>Test R²</th><th>MAE</th><th>Spearman</th></tr></thead><tbody>{[['Mean improvement',p.mean_baseline],['Baseline NLL alone',p.baseline_loss_only],['All pre-expert · ridge',p.ridge?.test],['All pre-expert · MLP',p.mlp?.test],['Target-free · ridge',d.target_free_probes.ridge?.test],['Target-free · MLP',d.target_free_probes.mlp?.test]].map(([name,s]:any)=><tr key={name}><td>{name}</td><td>{fmt(s?.r2)}</td><td>{fmt(s?.mae)}</td><td>{fmt(s?.spearman)}</td></tr>)}</tbody></table>
  <p>Success classifier: ROC-AUC {fmt(p.success_classifier?.test?.roc_auc)} · PR-AUC/AP {fmt(p.success_classifier?.test?.pr_auc)} · balanced accuracy {fmt(p.success_classifier?.test?.balanced_accuracy)} · prevalence {fmt(p.success_classifier?.test?.prevalence)}. Threshold fitted on analysis-train only.</p>
  <details><summary>Blind rankings, held-out interactions and probe details</summary><pre>{JSON.stringify(d,null,2)}</pre></details>
  <h3>Strength sweep · matched saved runs</h3><p>Only matching common hash, seed, budgets and measurement settings are included. No monotonic trend is assumed. {comparable.some(r=>r.calibration.strength===0)?'A matched 0% control is available below.':'Missing matched 0% control: detection verdicts remain provisional.'}</p>
  <table><thead><tr><th>Run</th><th>Strength</th><th>Task gap</th><th>Differential gap</th><th>Blind MLP R²</th><th>Target-free R²</th><th>Success AUC</th></tr></thead><tbody>{comparable.map(r=>{const v=r.calibration.experts[id];return <tr key={r.run_id}><td>{r.name}</td><td>{r.calibration.strength}</td><td>{fmt(v?.gap)}</td><td>{fmt(v?.differential_gap)}</td><td>{fmt(v?.mlp_r2)}</td><td>{fmt(v?.target_free_r2)}</td><td>{fmt(v?.auc)}</td></tr>;})}</tbody></table>
  <EChart replace option={{animation:false,tooltip:{trigger:'axis'},legend:{},xAxis:{type:'value',name:'Strength',min:0,max:1},yAxis:{type:'value',name:'Held-out Spearman',min:-1,max:1},series:current.map(k=>({type:'line',name:k,connectNulls:false,data:comparable.map(r=>[r.calibration.strength,r.calibration.experts[id]?.associations.find((v:Obj)=>v.feature===k)?.spearman??null])}))}}/>
  <h3>Reveal calibration labels</h3><button onClick={()=>setRevealed(!revealed)}>{revealed?'Hide calibration labels':'Reveal calibration labels'}</button>
  {revealed&&<><h4>Specialist × task · separate held-out sanity panel</h4><EChart option={matrix} replace/><p>{a.reveal[id].conclusion} (provisional). {a.reveal[id].warning}</p><pre>{JSON.stringify(a.reveal[id],null,2)}</pre><details><summary>Experimental controls and actual training distributions</summary><pre>{JSON.stringify(detail.specialization,null,2)}</pre></details></>}
 </section>;
}
