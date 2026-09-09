import {useState} from 'react';
import {EChart} from '../research/charts/EChart';
type Obj=Record<string,any>;
const fmt=(v:any)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(4):'—';

export function DiscoverySetup({detail,active,onRun}:{detail:Obj;active:boolean;onRun:(request:Obj)=>void}){
 const saved=detail.analysis?.discovery?.settings;
 const [limit,setLimit]=useState(saved?.max_features??5),[candidates,setCandidates]=useState(saved?.candidate_count??16),[cap,setCap]=useState(saved?.location_cap??20000),[seed,setSeed]=useState(saved?.seed??42);
 const source=detail.analysis?.discovery?.source_run_directory??detail.runDirectory;
 return <section className="panel"><h3>Find predictive features · saved observations</h3>
  <p>Uses this run’s observations.jsonl. No retraining, dataset download, checkpoint loading or expert probes. Fits diagnostic predictors of each expert’s improvement and differences between independent experts.</p>
  <fieldset disabled={active}><div className="lab-fields">
   <label>Maximum selected features<input type="number" min={1} max={10} value={limit} onChange={e=>setLimit(+e.target.value)}/></label>
   <label>Candidate feature count<input type="number" min={2} max={40} value={candidates} onChange={e=>setCandidates(+e.target.value)}/></label>
   <label>Analysis location cap<input type="number" min={100} max={100000} value={cap} onChange={e=>setCap(+e.target.value)}/></label>
   <label>Diagnostic seed<input type="number" min={0} value={seed} onChange={e=>setSeed(+e.target.value)}/></label>
  </div><p>Target-free pre-expert features only. Candidate screening uses fitting data; selection uses validation data; scores use held-out test data. Reusing inspected observations remains exploratory.</p>
  <div className="lab-actions"><button disabled={!source} onClick={()=>onRun({analysis_only:true,source_run_directory:source,max_features:limit,candidate_count:candidates,location_cap:cap,seed})}>Run saved-observation analysis</button></div></fieldset>
 </section>;
}

export function DiscoveryResults({detail}:{detail:Obj}){
 const result=detail.analysis.discovery,names=Object.keys(result.targets),[target,setTarget]=useState(names[0]??'');
 const d=result.targets[target];
 return <section className="panel"><h2>Predictive feature discovery</h2><p>{result.caveat}</p><p>{result.analyzed_locations} of {result.source_locations} source locations analyzed. {result.sampling}</p>
  <label>Outcome<select value={target} onChange={e=>setTarget(e.target.value)}>{names.map(n=><option key={n}>{n}</option>)}</select></label>
  <p>For pair advantage, positive means the first expert improves the same state more than the second. Identical experts have no differential signal to learn.</p>
  {d?.status==='ok'?<>
   <h3>Compact feature set</h3><p>{d.selected_features.join(', ')||'No feature improved validation sufficiently; mean baseline retained.'}</p>
   <p>Fitting / validation / test locations: {d.counts.train} / {d.counts.validation} / {d.counts.test}. Validation selected {d.validation_chosen_model}.</p>
   <table><thead><tr><th>Predictor</th><th>Test R²</th><th>Test MAE</th><th>Test Spearman</th></tr></thead><tbody>{Object.entries(d.scores).map(([k,v]:any)=><tr key={k}><td>{k.replaceAll('_',' ')}</td><td>{fmt(v.r2)}</td><td>{fmt(v.mae)}</td><td>{fmt(v.spearman)}</td></tr>)}</tbody></table>
   <h3>Which selected features matter?</h3><p>Remove each feature and refit. Positive MAE increase means removing it hurt prediction; an interval spanning zero is inconclusive. This measures predictive contribution, not causation.</p>
   <EChart replace option={{animation:false,tooltip:{},grid:{left:190,right:30},xAxis:{type:'value',name:'Test MAE increase after removal'},yAxis:{type:'category',data:d.ablations.map((v:Obj)=>v.feature)},series:[{type:'bar',data:d.ablations.map((v:Obj)=>v.mae_increase)}]}}/>
   <table><thead><tr><th>Removed feature</th><th>MAE increase</th><th>Paired 95% interval</th></tr></thead><tbody>{d.ablations.map((v:Obj)=><tr key={v.feature}><td>{v.feature}</td><td>{fmt(v.mae_increase)}</td><td>{v.mae_increase_ci.map(fmt).join(' … ')}</td></tr>)}</tbody></table>
   <h3>Selected feature relationships</h3><table><thead><tr><th>Feature</th><th>Test n</th><th>Spearman</th><th>95% interval</th></tr></thead><tbody>{d.held_out_associations.map((v:Obj)=><tr key={v.feature}><td>{v.feature}</td><td>{v.n}</td><td>{fmt(v.spearman)}</td><td>{v.interval?.map(fmt).join(' … ')??'—'}</td></tr>)}</tbody></table>
   <p>{d.verdict}</p><p>{d.note}</p><details><summary>Selection trace, redundant candidates and fitted model details</summary><pre>{JSON.stringify(d,null,2)}</pre></details>
  </>:<p>{d?.status??'No analyzable outcomes'}</p>}
  <h3>Saved analysis report</h3><pre className="lab-report">{detail.report}</pre><p>Saved as a separate run. Source observations and checkpoints remain unchanged. Share report.md and discovery-analysis.json for review.</p>
 </section>;
}
