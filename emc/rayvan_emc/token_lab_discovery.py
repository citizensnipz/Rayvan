"""Analysis of saved observations only. Never loads or executes an expert."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from .token_lab_features import blind_keys
from .token_lab_analysis import finite, corr, ranks, association
from .token_lab_calibration_analysis import split_rows, regression_scores


def validate(request):
    source=Path(request.get('source_run_directory',''))
    if not request.get('source_run_directory') or not (source/'observations.jsonl').is_file():
        raise ValueError('Choose a completed Token Lab run with observations.jsonl. Summary/display files cannot replace raw observations.')
    if not (source/'config.json').is_file():raise ValueError('Source config.json is missing')
    status=json.loads((source/'status.json').read_text())
    if status.get('status')!='completed':raise ValueError('Source run must be completed')
    for key,default,low,high in [('max_features',5,1,10),('candidate_count',16,2,40),('location_cap',20000,100,100000),('seed',42,0,2**31-1)]:
        v=request.get(key,default)
        if type(v)!=int or not low<=v<=high:raise ValueError(f'{key} must be an integer in [{low},{high}]')
    if request.get('max_features',5)>request.get('candidate_count',16):raise ValueError('Feature limit cannot exceed candidate count')
    return source


def read_locations(path,cap,seed,check,keep_task_metadata=False):
    """Stream grouped JSONL; retain uniformly sampled whole locations, not rows."""
    allowed=set(blind_keys(True))|{'sample_id','analysis_group','expert_id','topology','prefix_sha256','split','baseline_loss','expert_loss','improvement'}
    if keep_task_metadata:allowed.add('task_id') # Split/reveal metadata, never an input column.
    rng=random.Random(seed);bank=[];seen=0;sha=hashlib.sha256();current=[];last=None
    def add(group):
        nonlocal seen
        if not group:return
        if len({r['expert_id'] for r in group})!=len(group):raise ValueError('Duplicate expert rows in one location')
        seen+=1
        if len(bank)<cap:bank.append(group)
        else:
            j=rng.randrange(seen)
            if j<cap:bank[j]=group
    with path.open('rb') as f:
        for line in f:
            check();sha.update(line)
            if not line.strip():continue
            r=json.loads(line)
            if r.get('split') not in ['validation','evaluation']:raise ValueError('Only held-out evaluation observations are accepted')
            if any(not finite(r.get(k)) for k in ['baseline_loss','expert_loss','improvement']):raise ValueError('Nonfinite or missing expert outcomes')
            sid=r['sample_id']
            if last is not None and sid!=last:add(current);current=[]
            current.append({k:v for k,v in r.items() if k in allowed});last=sid
    add(current)
    return bank,seen,sha.hexdigest()


def targets(locations):
    result={}
    for group in locations:
        for r in group:result.setdefault(r['expert_id']+' improvement',[]).append(dict(r,value=r['improvement']))
        for a,b in itertools.combinations(sorted(group,key=lambda r:r['expert_id']),2):
            if a.get('topology')!='independent' or b.get('topology')!='independent':continue
            if not a.get('prefix_sha256') or a['prefix_sha256']!=b.get('prefix_sha256') or abs(a['baseline_loss']-b['baseline_loss'])>1e-6:
                raise ValueError('Pair analysis requires identical input prefix and common baseline')
            key=a['expert_id']+' minus '+b['expert_id']+' advantage'
            # Positive: A improves the SAME state more than B. Baseline cancels.
            result.setdefault(key,[]).append(dict(a,value=b['expert_loss']-a['expert_loss']))
    return result


def prepare(rows,split_masks=None):
    masks=split_rows(rows) if split_masks is None else split_masks;tr=masks<6;va=(masks>=6)&(masks<8);te=masks>=8
    if min(tr.sum(),va.sum(),te.sum())<15:return None
    keys=[k for k in blind_keys(True) if np.mean([finite(r.get(k)) for r,t in zip(rows,tr) if t])>=.8]
    X=np.array([[r.get(k) if finite(r.get(k)) else np.nan for k in keys] for r in rows],dtype=float)
    if not keys:return None
    median=np.nanmedian(X[tr],axis=0);X=np.where(np.isnan(X),median,X);mean=X[tr].mean(0);std=X[tr].std(0);keep=std>1e-8
    return (X[:,keep]-mean[keep])/std[keep],np.array([r['value'] for r in rows]),[k for k,kp in zip(keys,keep) if kp],(tr,va,te)


def fit(X,y,masks,indices,kind,seed,check):
    tr,va,te=masks;center=y[tr].mean()
    if not indices:return np.full(len(y),center),dict(kind='training mean')
    x=X[:,indices]
    if kind=='ridge':
        best=None
        for alpha in [1.,10.,100.,1000.]:
            check();w=np.linalg.solve(x[tr].T@x[tr]+alpha*np.eye(len(indices)),x[tr].T@(y[tr]-center));p=x@w+center
            candidate=(float(np.mean((p[va]-y[va])**2)),p,dict(kind='ridge',alpha=alpha,coefficients=w.tolist()))
            if best is None or candidate[0]<best[0]:best=candidate
        return best[1],best[2]
    scale=max(float(y[tr].std()),1e-6);xt=torch.tensor(x,dtype=torch.float32);yt=torch.tensor((y-center)/scale,dtype=torch.float32)
    torch.manual_seed(seed);net=nn.Sequential(nn.Linear(len(indices),32),nn.GELU(),nn.Linear(32,1))
    optimizer=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.01);best=float('inf');state=None;best_epoch=0
    for epoch in range(150):
        check();optimizer.zero_grad();p=net(xt[tr]).squeeze(-1);loss=nn.functional.mse_loss(p,yt[tr]);loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(),1);optimizer.step()
        with torch.no_grad():score=float(nn.functional.mse_loss(net(xt[va]).squeeze(-1),yt[va]))
        if score<best:best=score;state=copy.deepcopy(net.state_dict());best_epoch=epoch
        if epoch-best_epoch>=20:break
    net.load_state_dict(state)
    with torch.no_grad():p=net(xt).squeeze(-1).numpy()*scale+center
    return p,dict(kind='MLP',hidden_units=32,epochs=best_epoch+1)


def select(X,y,keys,masks,limit,candidates,seed,check):
    tr,va,te=masks
    # Training-only rank screen, then prune near-duplicate measured quantities.
    order=sorted(range(len(keys)),key=lambda j:(-abs(corr(ranks(X[tr,j]),ranks(y[tr])) or 0),keys[j]))
    pool=[];duplicates={}
    for j in order:
        duplicate=next((k for k in pool if abs(corr(X[tr,j],X[tr,k]) or 0)>.98),None)
        if duplicate is not None:duplicates[keys[j]]=keys[duplicate];continue
        pool.append(j)
        if len(pool)>=candidates:break
    selected=[];trace=[];best=float(np.mean((y[va]-y[tr].mean())**2))
    for _ in range(limit):
        trials=[]
        for j in pool:
            if j in selected:continue
            p,meta=fit(X,y,masks,selected+[j],'ridge',seed,check)
            trials.append((float(np.mean((p[va]-y[va])**2)),j))
        if not trials:break
        loss,j=min(trials,key=lambda v:(v[0],keys[v[1]]))
        if loss>=best-max(1e-10,.005*best):break
        selected.append(j);best=loss;trace.append(dict(added=keys[j],validation_mse=loss))
    return selected,dict(candidates=[keys[j] for j in pool],near_duplicates=duplicates,trace=trace,
        method='train Spearman top candidates, |Pearson|>.98 redundancy pruning; greedy validation ridge MSE; at least .5% relative improvement to add a feature; test never used for selection')


def paired_interval(rows,delta,seed):
    # Group bootstrap: resample generated examples rather than related endpoints.
    groups={}
    for i,r in enumerate(rows):groups.setdefault(r.get('analysis_group',r['sample_id']),[]).append(i)
    groups=list(groups.values());rng=np.random.default_rng(seed);values=[]
    for _ in range(200):
        idx=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))]);values.append(float(delta[idx].mean()))
    return np.quantile(values,[.025,.975]).tolist()


def analyze_target(rows,request,check):
    data=prepare(rows)
    if data is None:return dict(status='insufficient available inputs or split sizes')
    X,y,keys,masks=data;tr,va,te=masks
    if np.var(y[tr])<1e-12 and np.var(y[te])<1e-12:
        return dict(status='constant outcome; no state-dependent difference to explain',n=len(rows),constant=float(y[tr].mean()))
    seed=request.get('seed',42);selected,selection=select(X,y,keys,masks,request.get('max_features',5),request.get('candidate_count',16),seed,check)
    scores={};predictions={};fits={}
    for name,idx,kind in [('mean',[],'ridge'),('all_ridge',list(range(len(keys))),'ridge'),('all_mlp',list(range(len(keys))),'mlp'),('compact_ridge',selected,'ridge'),('compact_mlp',selected,'mlp')]:
        p,meta=fit(X,y,masks,idx,kind,seed,check);scores[name]=regression_scores(y[te],p[te]);predictions[name]=p;fits[name]=meta
    # Choose diagnostic model type using validation ONLY, not the test scores.
    winner=min(['compact_ridge','compact_mlp'],key=lambda name:np.mean((predictions[name][va]-y[va])**2))
    kind='ridge' if winner.endswith('ridge') else 'mlp';full=predictions[winner][te];testrows=[r for r,t in zip(rows,te) if t]
    ablations=[]
    for j in selected:
        check();p,_=fit(X,y,masks,[k for k in selected if k!=j],kind,seed,check)
        difference=abs(p[te]-y[te])-abs(full-y[te])
        ablations.append(dict(feature=keys[j],test=regression_scores(y[te],p[te]),mae_increase=float(difference.mean()),
            mae_increase_ci=paired_interval(testrows,difference,seed)))
    associations=[association(testrows,keys[j],'value') for j in selected]
    beats=scores[winner]['mae']<scores['mean']['mae']
    return dict(status='ok',counts=dict(train=int(tr.sum()),validation=int(va.sum()),test=int(te.sum())),
        available_features=keys,selected_features=[keys[j] for j in selected],selection=selection,scores=scores,fits=fits,
        validation_chosen_model=winner,ablations=ablations,held_out_associations=associations,
        verdict='Predictive on this held-out split; exploratory, replicate on new observations.' if beats else 'Compact features did not beat the mean baseline on held-out data.',
        note='Positive ablation MAE increase means removing the feature hurt prediction. Refit effects include training variability and shared information; they are not causal importance. Ridge screening can miss purely nonlinear interactions; inspect all_mlp versus compact_mlp.')


def report(result):
    lines=['# Token Lab — saved feature discovery',result['caveat'],
        f"Source: {result['source_run_directory']}; raw SHA256: {result['raw_sha256']}",
        f"Locations analyzed: {result['analyzed_locations']} / {result['source_locations']}. {result['sampling']}",
        'Target-free pre-expert features only. Correct-target probabilities/NLL/margins, task labels, specialist assignments, history, expert internals and post-expert features are excluded.',
        'Positive pair advantage means the first expert improves the same state more than the second. Serial stages are analyzed individually; no cross-stage counterfactual claim.',
        'The existing example-group 60/20/20 split is preserved. Feature screening uses train, subset/model selection uses validation, reported scores use test. Reusing previously inspected test observations makes this exploratory rather than independent confirmation.']
    for name,d in result['targets'].items():
        lines+=['',f'## {name}']
        if d['status']!='ok':lines.append(d['status']);continue
        lines+=['Selected features: '+(', '.join(d['selected_features']) or 'none'),
            f"Split counts: {d['counts']}; validation-chosen compact model: {d['validation_chosen_model']}",
            '| Predictor | Test R² | Test MAE | Test Spearman |','|---|---:|---:|---:|']
        for k,s in d['scores'].items():lines.append(f"| {k} | {s['r2']} | {s['mae']} | {s['spearman']} |")
        lines+=['','### Removing one feature and refitting','| Removed feature | Test MAE increase | Paired 95% interval |','|---|---:|---|']
        for v in d['ablations']:lines.append(f"| {v['feature']} | {v['mae_increase']} | {v['mae_increase_ci']} |")
        lines+=['','### Selected feature associations on test observations']
        for v in d['held_out_associations']:lines.append(f"- {v['feature']}: Spearman {v['spearman']}; interval {v['interval']}; n={v['n']}")
        lines += [d['verdict'],d['note']]
    return '\n'.join(lines)


def run_saved(request,runs_root,run_id,cancel_path=None):
    source=validate(request)
    if not run_id or any(not (v.isascii() and (v.isalnum() or v in '-_')) for v in run_id):raise ValueError('Invalid run ID')
    root=Path(runs_root)/run_id;root.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    def write(name,obj):(root/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    def check():
        if (root/'cancel.requested').exists() or (cancel_path and Path(cancel_path).exists()):raise InterruptedError('Cancelled saved analysis')
    def emit(phase,**kw):
        ev=dict(type='token_lab_progress',run_id=run_id,phase=phase,elapsed_seconds=time.perf_counter()-start,**kw)
        write('status.json',dict(status='running',**ev));print(json.dumps(ev),flush=True)
        with (root/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(ev)+'\n')
    try:
        config=json.loads((source/'config.json').read_text());write('config.json',config);write('discovery-config.json',request)
        torch.set_num_threads(1);emit('reading saved observations')
        locations,total,sha=read_locations(source/'observations.jsonl',request.get('location_cap',20000),request.get('seed',42),check)
        result=dict(schema_version=1,source_run_directory=str(source.resolve()),raw_sha256=sha,source_locations=total,settings=request,
            analyzed_locations=len(locations),sampling='Uniform reservoir of whole locations if cap is exceeded; all rows otherwise.',
            caveat='Observational diagnostic models only. No experts, checkpoints, dataset generation or router are executed. No multiplicity-adjusted claims.',targets={})
        datasets=targets(locations)
        for i,(name,rows) in enumerate(datasets.items()):
            check();emit('finding predictive features',step=i+1,total=len(datasets),expert_id=name,measured_locations=len(locations))
            result['targets'][name]=analyze_target(rows,request,check)
        check();write('discovery-analysis.json',result);write('analysis.json',dict(discovery=result))
        (root/'report.md').write_text(report(result),encoding='utf-8')
        summary=dict(run_id=run_id,name=config.get('name','Token Lab')+' · feature discovery',experiment_type='token_lab_discovery',
            status='completed',started_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),measured_locations=len(locations),
            observations=sum(map(len,locations)),analysis_locations=len(locations),training_targets=0,runtime_seconds=time.perf_counter()-start,
            source_run_directory=str(source.resolve()),raw_sha256=sha)
        write('summary.json',summary);emit('complete');write('status.json',dict(status='completed'));return summary
    except BaseException as exc:
        write('status.json',dict(status='cancelled' if isinstance(exc,InterruptedError) else 'failed',error=str(exc)));raise
