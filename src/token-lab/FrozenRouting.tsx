import {useEffect,useState} from 'react';
import {invoke} from '@tauri-apps/api/core';
import {EChart} from '../research/charts/EChart';
type Obj=Record<string,any>;
export const routingDefaults={experiment_mode:'frozen_routing',name:'Frozen Expert Routing Test',source_sweep:'',
 family:'gpt',source_seed:'all',include_control:true,router_seed:142,training_samples:1000,validation_samples:300,
 test_samples:1000,rounds:3,fit_epochs:80,learning_rate:.001,exploration:.2,sequential_steps:3,spectral_enabled:false,device:'cpu'};
const fmt=(v:any)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(4):'—';

export function RoutingSetup({config,setConfig,runs}:{config:Obj;setConfig:(v:any)=>void;runs:Obj[]}){
 const [source,setSource]=useState<Obj>(),[error,setError]=useState('');
 const available=runs.filter(r=>r.experiment_type==='token_lab_sweep');
 const chosen=available.find(r=>r.runDirectory===config.source_sweep);
 useEffect(()=>{
  let cancelled=false;setSource(undefined);setError('');
  if(chosen)void invoke<Obj>('get_token_lab',{runId:chosen.run_id}).then(d=>{if(!cancelled)setSource(d);}).catch(e=>{if(!cancelled)setError(String(e));});
  return()=>{cancelled=true;};
 },[chosen?.run_id,config.source_sweep]);
 const set=(k:string,v:any)=>setConfig((c:Obj)=>({...c,[k]:v}));
 const entries:Obj[]=source?.analysis?.sweep?.entries??[];
 const selected=entries.filter(e=>(config.family==='all'||e.family===config.family)&&(String(config.source_seed)==='all'||e.seed===Number(config.source_seed))&&(config.include_control||e.strength===1));
 const field=(key:string,label:string,min=1,step=1)=><label key={key}>{label}<input type="number" min={min} step={step} value={config[key]} onChange={e=>set(key,+e.target.value)}/></label>;
 return <section className="panel" style={{gridColumn:'1 / -1'}}><h3>Frozen Expert Routing Test</h3>
  <p>Load trained specialists directly from your local validation sweep. No common-base or specialist training; checkpoint dimensions, tokenizer and reference bank are restored automatically.</p>
  <div className="lab-fields">
   <label>Test name<input value={config.name} onChange={e=>set('name',e.target.value)}/></label>
   <label>Source validation sweep<select aria-label="Source validation sweep" value={chosen?.run_id??''} onChange={e=>{const r=available.find(r=>r.run_id===e.target.value);setConfig((c:Obj)=>({...c,source_sweep:r?.runDirectory??'',source_seed:'all'}));}}><option value="">Choose your completed sweep</option>{available.map(r=><option key={r.run_id} value={r.run_id}>{r.name} · {r.run_id} · {r.completed_jobs}/{r.planned_jobs} · {r.status}</option>)}</select></label>
   <label>Expert family<select value={config.family} onChange={e=>set('family',e.target.value)}>{['gpt','ssm','recurrent','delta','all'].map(f=><option key={f} value={f}>{f==='all'?'All four families':f==='recurrent'?'Recurrent / GRU':f.toUpperCase()}</option>)}</select></label>
   <label>Saved training seed<select value={config.source_seed} onChange={e=>set('source_seed',e.target.value==='all'?'all':+e.target.value)}><option value="all">All seeds in sweep</option>{[...new Set(entries.map(e=>e.seed))].map(seed=><option key={seed}>{seed}</option>)}</select></label>
   <label><input type="checkbox" checked={config.include_control} onChange={e=>set('include_control',e.target.checked)}/> Include paired 0% controls (100% always included)</label>
   <label>Device<select value={config.device} onChange={e=>set('device',e.target.value)}><option>cpu</option><option>cuda</option></select></label>
  </div>
  <p>Automatic comparison: <strong>one step and {config.sequential_steps} sequential steps; simple features and full descriptors</strong>. Baselines: training-selected fixed expert sequence, exact expected uniform-random sequence loss, and a diagnostic oracle. Experts and shared machinery stay frozen.</p>
  <p>{source?`${selected.length} selected checkpoint banks (${selected.filter(e=>e.status==='completed').length} completed), ${selected.length*4} routing conditions.`:'Choose a sweep to preview the saved checkpoints.'} Router learns only from fresh training probes. Validation selects updates; untouched test data scores final executed loss.</p>
  {selected.length>0&&<details><summary>Checkpoints that will be tested</summary><ul>{selected.map(e=><li key={e.key}>{e.family.toUpperCase()} · seed {e.seed} · {e.strength*100}% · {e.status}</li>)}</ul></details>}
  <details><summary>Router training and test budgets</summary><div className="lab-fields">
   {field('training_samples','Fresh router training locations',100,10)}{field('validation_samples','Validation locations',100,10)}{field('test_samples','Final test locations',100,10)}
   {field('rounds','Counterfactual policy rounds')}{field('fit_epochs','Preference fitting epochs')}{field('router_seed','Router / fresh data seed',0)}{field('learning_rate','Router learning rate',.000001,.0001)}{field('exploration','Training exploration probability',0,.05)}
   <label>Sequential steps<select value={config.sequential_steps} onChange={e=>set('sequential_steps',+e.target.value)}><option>2</option><option>3</option></select></label>
   <label><input type="checkbox" checked={config.spectral_enabled} onChange={e=>set('spectral_enabled',e.target.checked)}/> Add spectra to full descriptors (slower)</label>
  </div></details>
  <p>Defaults: 1000 training, 300 validation, 1000 test locations; 3 policy rounds. Full descriptors use state geometry and saved training-reference novelty. Task labels, target probabilities and post-expert measurements are forbidden inputs. Spectra start off.</p>
  <p>No production EMC router is replaced. This uses the saved Token Lab residual expert application. Repeated application is a new test: these specialists were originally trained for one step.</p>
  {error&&<p role="alert">{error}</p>}
 </section>;
}

export function RoutingResults({detail}:{detail:Obj}){
 const a=detail.analysis.routing_test;
 const rows=(a.entries??[]).flatMap((e:Obj)=>(e.results??[]).map((r:Obj)=>({...r,source:e.key})));
 return <section className="panel"><h2>Frozen expert routing results</h2>
  <p>{a.status} · {a.entries.filter((e:Obj)=>e.status==='completed').length}/{a.planned_jobs} saved expert banks tested. Positive gain means lower executed loss than the baseline.</p>
  <EChart replace option={{animation:false,tooltip:{trigger:'axis'},legend:{},grid:{left:65,bottom:120},xAxis:{type:'category',data:rows.map((r:Obj)=>`${r.source} / ${r.horizon} / ${r.features}`),axisLabel:{rotate:50,fontSize:9}},yAxis:{type:'value',name:'Loss reduction (nats)'},series:[{name:'Gain vs fixed sequence',type:'bar',data:rows.map((r:Obj)=>r.comparisons.fixed_sequence.gain)},{name:'Gain vs random',type:'bar',data:rows.map((r:Obj)=>r.comparisons.uniform_random.gain)}]}}/>
  <h3>Replication across saved seeds</h3><table><thead><tr><th>Condition</th><th>Mean gain vs fixed</th><th>Positive seeds</th><th>Positive intervals</th><th>Learned policies</th></tr></thead><tbody>{Object.entries(a.replication??{}).map(([key,g]:any)=><tr key={key}><td>{key}</td><td>{fmt(g.mean_gain)}</td><td>{g.positive_seeds}/{g.replicates}</td><td>{g.positive_intervals}/{g.replicates}</td><td>{g.learned_policies}/{g.replicates}</td></tr>)}</tbody></table>
  <h3>Individual conditions</h3><div className="lab-table"><table><thead><tr><th>Source</th><th>Steps</th><th>Inputs</th><th>Policy</th><th>Test NLL</th><th>Gain vs fixed</th><th>95% interval</th><th>Gain vs random</th><th>Gain vs simple</th><th>Suffix regret</th><th>Endpoint/s</th><th>Context tok/s</th></tr></thead><tbody>{rows.map((r:Obj)=><tr key={`${r.source}-${r.horizon}-${r.features}`}><td>{r.source}</td><td>{r.horizon}</td><td>{r.features}</td><td>{r.selected_policy}</td><td>{fmt(r.mean_loss)}</td><td>{fmt(r.comparisons.fixed_sequence.gain)}</td><td>{r.comparisons.fixed_sequence.gain_ci.map(fmt).join(' … ')}</td><td>{fmt(r.comparisons.uniform_random.gain)}</td><td>{fmt(r.comparisons.simple_router?.gain)}</td><td>{fmt(r.suffix_regret)}</td><td>{fmt(r.endpoints_per_second)}</td><td>{fmt(r.context_tokens_per_second)}</td></tr>)}</tbody></table></div>
  <p>Context tok/s counts prefix tokens once per endpoint. It is not generated tok/s. Timings exclude counterfactual audits. Confidence intervals are pointwise and exploratory. The fixed baseline may be retained if validation rejects every learned candidate.</p>
  {(a.entries??[]).filter((e:Obj)=>e.status!=='completed').map((e:Obj)=><p role="alert" key={e.key}>{e.key}: {e.status} — {e.error}</p>)}
  <details><summary>Validation rounds, selected features, expert usage and frozen-model checks</summary><pre>{JSON.stringify(a.entries,null,2)}</pre></details>
  <h3>Report</h3><p>Share report.md and routing-test-analysis.json. Checkpoints, counterfactual targets and individual test decisions are saved under jobs/.</p><pre className="lab-report">{detail.report}</pre>
 </section>;
}

export function RoutingProgress({events,progress}:{events:Obj[];progress?:Obj}){
 const latest=events.at(-1)?.phase;const points=events.filter(e=>e.phase===latest);
 return <section className="panel"><h3>Router validation during training</h3><p>{progress?.phase??'Preparing saved checkpoints'}</p>
  <p>Validation guides acceptance. Final test loss is measured only after router training finishes; it is not used to select a round.</p>
  {events.length?<><p>{latest}</p><EChart replace option={{animation:false,tooltip:{trigger:'axis'},xAxis:{type:'category',data:points.map((e:Obj)=>e.step),name:'Policy round'},yAxis:{type:'value',name:'Candidate validation NLL',scale:true},series:[{type:'line',data:points.map((e:Obj)=>e.loss)}]}}/>
   <table><thead><tr><th>Condition</th><th>Round</th><th>Validation NLL</th><th>Accepted?</th></tr></thead><tbody>{events.slice(-12).map((e:Obj,i:number)=><tr key={i}><td>{e.phase}</td><td>{e.step}</td><td>{fmt(e.loss)}</td><td>{e.accepted?'Yes':'No — kept previous policy'}</td></tr>)}</tbody></table><p>The chart shows only the most recently validated condition. The table retains recent updates across conditions.</p></>:<p>Collecting initial counterfactual probes. The first validation point appears after fitting the first candidate.</p>}
 </section>;
}
