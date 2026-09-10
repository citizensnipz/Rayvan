"""Frozen-expert routing intervention, separate from observational Token Lab modes.

Counterfactual policy iteration: collect same-state, fixed-continuation suffix
losses; fit a regret-weighted pairwise classifier; accept on validation final
loss; execute once on untouched test inputs. No expert/shared parameter updates.
"""
from dataclasses import replace
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .token_lab import LabConfig, Dataset, ProbeModel, write
from .token_lab_calibration import CalibrationDataset, digest
from .token_lab_discovery import paired_interval
from .token_lab_sweep import load, checkpoint_json
from . import token_lab_features as mf
from .capability_tasks import CAPABILITIES

DEFAULTS = dict(experiment_mode='frozen_routing', name='Frozen Expert Routing Test',
    source_sweep='', family='gpt', source_seed='all', include_control=True,
    router_seed=142, training_samples=1000, validation_samples=300, test_samples=1000,
    rounds=3, fit_epochs=80, learning_rate=.001, exploration=.2,
    sequential_steps=3, spectral_enabled=False, device='cpu')


def sources(c):
    root=Path(c['source_sweep'])
    sweep=load(root/'sweep-analysis.json')
    if sweep.get('schema_version')!=1:raise ValueError('Unsupported sweep schema')
    rows=[e for e in sweep['entries'] if (c['family']=='all' or e['family']==c['family'])
          and (str(c['source_seed'])=='all' or e['seed']==int(c['source_seed']))
          and (c['include_control'] or e['strength']==1)]
    if not rows:raise ValueError('No matching sweep experts. Choose a family/seed present in this sweep.')
    if any(e['status']!='completed' for e in rows):raise ValueError('Selected sweep contains unfinished jobs. Select a completed family/seed.')
    if c['include_control']:
        for e in rows:
            other=next((v for v in rows if v['family']==e['family'] and v['seed']==e['seed'] and v['strength']==1-e['strength']),None)
            if other is None:raise ValueError('A matching 0%/100% checkpoint pair is missing')
            if other['controls']['common_hash']!=e['controls']['common_hash']:raise ValueError('0%/100% pair does not share the same common base')
    result=[]
    for e in rows:
        expected=f"{e['family']}-seed-{e['seed']}-s{int(e['strength']*100)}"
        if e['key']!=expected or e['family'] not in ['gpt','ssm','recurrent','delta'] or type(e['seed']) is not int or e['strength'] not in [0,1]:raise ValueError('Invalid sweep job identity')
        local=root/'jobs'/e['key']
        path=local if (local/'model.pt').is_file() else Path(e['source_directory'])
        for file in ['model.pt','config.json','specialization.json','reference-bank.pt','observations.jsonl']:
            if not (path/file).is_file():raise ValueError(f'{e["key"]}: missing {file}. Keep the complete local sweep jobs directory; report files alone cannot load experts.')
        cfg=load(path/'config.json');info=load(path/'specialization.json')
        if cfg['experiment_mode']!='forced_specialization' or len(cfg['families'])!=2 or len(set(cfg['families']))!=1:raise ValueError('Two homogeneous saved specialists required')
        if cfg['families'][0]!=e['family'] or cfg['seed']!=e['seed'] or info['strength']!=e['strength']:raise ValueError('Sweep/checkpoint provenance mismatch')
        if info['common_hash']!=e['controls']['common_hash']:raise ValueError('Sweep common-base hash mismatch')
        if not info['clone_identity_verified'] or not info['shared_unchanged']:raise ValueError('Saved calibration controls failed')
        if e['strength']==0 and not info['exact_control_final_identity']:raise ValueError('Saved 0% control is not an identical-expert control')
        result.append((e,path.resolve()))
    return result


def validate(request):
    c={k:request.get(k,v) for k,v in DEFAULTS.items()}
    if not c['source_sweep']:raise ValueError('Choose a saved Specialization Validation Sweep')
    if c['family'] not in ['all','gpt','ssm','recurrent','delta']:raise ValueError('Unsupported family')
    if str(c['source_seed'])!='all':
        try:c['source_seed']=int(c['source_seed'])
        except (TypeError,ValueError):raise ValueError('Source seed must be all or an integer')
    for key,lo,hi in [('router_seed',0,2**31-1),('rounds',1,10),('fit_epochs',1,500),('sequential_steps',2,3),
                       ('training_samples',100,10000),('validation_samples',100,5000),('test_samples',100,10000)]:
        if type(c[key]) is not int or not lo<=c[key]<=hi:raise ValueError(f'{key} must be an integer in [{lo},{hi}]')
    for k in ['training_samples','validation_samples','test_samples']:
        if c[k]%10:raise ValueError(k+' must be divisible by 10 for balanced tasks')
    if not math.isfinite(c['learning_rate']) or not 0<c['learning_rate']<=.1:raise ValueError('Invalid learning rate')
    if not math.isfinite(c['exploration']) or not 0<=c['exploration']<=1:raise ValueError('Exploration must be in [0,1]')
    if any(type(c[k]) is not bool for k in ['include_control','spectral_enabled']):raise ValueError('Invalid toggle')
    if c['device'] not in ['cpu','cuda'] or (c['device']=='cuda' and not torch.cuda.is_available()):raise ValueError('Selected device unavailable')
    sources(c)
    return c


def load_model(path,device):
    saved=torch.load(path/'model.pt',map_location='cpu',weights_only=True)
    cfg=LabConfig(**(saved['config']|dict(device=device))).validate()
    if load(path/'config.json')!=json.loads(json.dumps(saved['config'])):raise ValueError('Model/config mismatch')
    ds=Dataset(cfg);model=ProbeModel(cfg,ds.tokenizer.vocab_size)
    if saved['tokenizer']!=ds.tokenizer.to_config():raise ValueError('Tokenizer mismatch')
    model.load_state_dict(saved['model'],strict=True)
    info=load(path/'specialization.json')
    if [digest(e.state_dict()) for e in model.experts]!=info['final_hashes']:raise ValueError('Expert hash mismatch')
    shared={k:v for k,v in model.state_dict().items() if not k.startswith('experts.')}
    if digest(shared)!=info['shared_hash']:raise ValueError('Shared machinery hash mismatch')
    model.to(device).eval().requires_grad_(False)
    ref=torch.load(path/'reference-bank.pt',map_location='cpu',weights_only=True)
    # Historical expert success is forbidden. Common pre-expert novelty bank only.
    bank={k:v for k,v in ref['banks'][0].items() if k!='success'}
    excluded=set(saved['excluded_prefixes'])|set(ref['excluded_training_prefix_hashes'])
    groups=set()
    with (path/'observations.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);excluded.add(r['prefix_sha256']);groups.add(r.get('analysis_group',r['sample_id']))
    return cfg,ds,model,bank,excluded,groups,info


def make_pools(c,cfg,base_ds,excluded,previous_groups,check):
    # Keep checkpoint suite/tokenizer. Router seed controls new streams, never model initialization.
    ds=CalibrationDataset(replace(cfg,seed=c['router_seed']),base_ds);ds.excluded=set(excluded)
    owner={g:'source' for g in previous_groups};result={}
    for split,n in [('train',c['training_samples']),('validation',c['validation_samples']),('test',c['test_samples'])]:
        rows=[]
        for i in range(n):
            check();task=CAPABILITIES[i%10]
            for attempt in range(10000):
                x,y,m=ds.sample_task('validation' if split=='train' else 'evaluation',i+attempt*100003,task,
                    stream={'train':801,'validation':802,'test':803}[split])
                group=m['analysis_group']
                if group in owner:continue
                owner[group]=split;m['router_split']=split
                rows.append((x,y,m));break
            else:raise ValueError('Unable to obtain disjoint router samples; reduce sample budget')
        result[split]=rows
    return result


class Observer:
    """Target/task-blind API. Only prefix tokens, causal position and current state."""
    def __init__(self,model,cfg,ds,bank,spectral=False):
        self.model=model;self.cfg=cfg;self.counts=ds.counts;self.bank=bank;self.spectral=spectral
        self.seconds=0.;self.calls=0
    @torch.no_grad()
    def features(self,h,x,source_position,remaining,horizon,kind):
        start=time.perf_counter();logits=self.model.logits(h)[0].float();lp=logits.log_softmax(-1);p=lp.exp()
        token=int(x[0,-1]);freq=(int(self.counts[token])+1)/(int(self.counts.sum())+len(self.counts))
        row=dict(position=x.shape[1]-1,position_normalized=(x.shape[1]-1)/self.cfg.sequence_length,
            source_position=source_position,context_start=source_position+1-x.shape[1],
            token_count=int(self.counts[token]),token_frequency=freq,log_token_frequency=math.log(freq),rarity=-math.log(freq),token_char_length=1,
            predictive_entropy=float(-(p*lp).sum()),top1_probability=float(p.max()),logit_margin=float(logits.topk(2).values.diff().neg()[0]))
        if kind=='full':
            row.update(mf.core(h[0,-self.cfg.window:],self.cfg.knn_k))
            row.update(mf.novelty(h[0,-1],row,self.bank,self.cfg.knn_k))
            if self.spectral:row.update(mf.spectral(h[0,-self.cfg.window:]))
        allowed=set(mf.blind_keys(True))
        row={k:v for k,v in row.items() if k in allowed}
        row.update(remaining_steps=remaining,trajectory_step=horizon-remaining)
        self.seconds+=time.perf_counter()-start;self.calls+=1
        return row


class Policy:
    def __init__(self,sequence,keys=(),median=None,scale=None,net=None):
        self.sequence=tuple(sequence);self.keys=list(keys);self.median=median;self.scale=scale;self.net=net
    def choose(self,observer,h,x,pos,remaining,kind):
        if self.net is None:return self.sequence[len(self.sequence)-remaining]
        row=observer.features(h,x,pos,remaining,len(self.sequence),kind)
        a=np.array([row.get(k) if row.get(k) is not None else np.nan for k in self.keys],dtype=np.float32)
        a=np.where(np.isfinite(a),a,self.median)
        z=torch.from_numpy(np.clip((a-self.median)/self.scale,-20,20)).float()[None]
        with torch.no_grad():score=float(self.net(z)[0,0])
        return 0 if score>0 else 1
    @classmethod
    def load(cls,path):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['network'] is None:return cls(saved['sequence'])
        net=nn.Sequential(nn.Linear(len(saved['keys']),32),nn.GELU(),nn.Linear(32,1))
        net.load_state_dict(saved['network'],strict=True)
        return cls(saved['sequence'],saved['keys'],saved['median'].numpy(),saved['scale'].numpy(),net.eval().requires_grad_(False))
    def save(self,path):
        torch.save(dict(sequence=self.sequence,keys=self.keys,
            median=None if self.median is None else torch.tensor(self.median),
            scale=None if self.scale is None else torch.tensor(self.scale),
            network=None if self.net is None else self.net.state_dict(),hidden=32),path)


def preference_loss(logits,advantages):
    """|Q_B-Q_A| log(1+exp(-sign(Q_B-Q_A)*logit)); zero weight for ties."""
    return (advantages.abs()*F.softplus(-advantages.sign()*logits)).mean()


def fit_policy(rows,advantages,sequence,c,seed,check):
    a=np.asarray(advantages,dtype=np.float32)
    if np.max(abs(a))<1e-8:return Policy(sequence)
    keys=sorted(set().union(*(r.keys() for r in rows)))
    X=np.array([[r.get(k) if r.get(k) is not None else np.nan for k in keys] for r in rows],dtype=np.float32)
    valid=np.isfinite(X).mean(0)>=.8;X=X[:,valid];keys=[k for k,v in zip(keys,valid) if v]
    median=np.nanmedian(np.where(np.isfinite(X),X,np.nan),axis=0)
    X=np.where(np.isfinite(X),X,median);scale=X.std(0);valid=scale>1e-6
    X=X[:,valid];median=median[valid];scale=scale[valid];keys=[k for k,v in zip(keys,valid) if v]
    # A bias-only network is a valid fitted constant preference.
    Z=torch.from_numpy(np.clip((X-median)/scale,-20,20)).float()
    y=torch.from_numpy(a/max(float(abs(a).mean()),1e-8))
    torch.manual_seed(seed);net=nn.Sequential(nn.Linear(len(keys),32),nn.GELU(),nn.Linear(32,1))
    opt=torch.optim.AdamW(net.parameters(),lr=c['learning_rate'],weight_decay=.01)
    gen=torch.Generator().manual_seed(seed)
    for _ in range(c['fit_epochs']):
        check()
        for indices in torch.randperm(len(Z),generator=gen).split(128):
            opt.zero_grad(set_to_none=True);loss=preference_loss(net(Z[indices])[:,0],y[indices]);loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
    return Policy(sequence,keys,median,scale,net.eval().requires_grad_(False))


@torch.no_grad()
def rollout(model,h,x,pos,remaining,policy,observer,kind,check,first=None):
    choices=[]
    for r in range(remaining,0,-1):
        check();e=first if r==remaining and first is not None else policy.choose(observer,h,x,pos,r,kind)
        h=h+model.experts[e](h);choices.append(e)
    return h,choices


def nll(model,h,target):
    return float(F.cross_entropy(model.logits(h),torch.tensor([target],device=h.device)))


@torch.no_grad()
def path_losses(model,pool,horizon,check):
    """Diagnostic exhaustive tree; never called by a learned route to choose its expert."""
    seqs=list(itertools.product(range(2),repeat=horizon));losses=[]
    for x,y,m in pool:
        check();h=model.embed(x.to(next(model.parameters()).device));out=[]
        def walk(state,depth):
            check()
            if depth==horizon:out.append(nll(model,state,y));return
            for expert in model.experts:walk(state+expert(state),depth+1)
        walk(h,0);losses.append(out)
    return seqs,np.asarray(losses)


@torch.no_grad()
def collect(model,pool,horizon,teacher,observer,kind,c,seed,check,file):
    rows=[];ys=[];rng=random.Random(seed)
    for x,y,m in pool:
        x=x.to(next(model.parameters()).device);h=model.embed(x)
        for remaining in range(horizon,0,-1):
            check();row=observer.features(h,x,m['source_position'],remaining,horizon,kind);q=[]
            for e in range(2):
                after,_=rollout(model,h,x,m['source_position'],remaining,teacher,observer,kind,check,first=e)
                q.append(nll(model,after,y))
            rows.append(row);ys.append(q[1]-q[0])
            e=rng.randrange(2) if rng.random()<c['exploration'] else teacher.choose(observer,h,x,m['source_position'],remaining,kind)
            file.write(json.dumps(dict(sample_id=m['sample_id'],analysis_group=m['analysis_group'],remaining_steps=remaining,
                features=row,suffix_losses=q,advantage=q[1]-q[0],behaviour_choice=e))+'\n')
            h=h+model.experts[e](h)
    return rows,ys


@torch.no_grad()
def execute(model,pool,horizon,policy,observer,kind,check,audit=False,file=None):
    losses=[];regrets=[];routes=[];started=time.perf_counter();feature_start=observer.seconds;audit_seconds=0.
    for x,y,m in pool:
        check();x=x.to(next(model.parameters()).device);h=model.embed(x);chosen=[];sample_audit=[]
        for remaining in range(horizon,0,-1):
            e=policy.choose(observer,h,x,m['source_position'],remaining,kind)
            # Execute selected expert BEFORE any diagnostic alternatives.
            after=h+model.experts[e](h);chosen.append(e)
            if audit:
                t=time.perf_counter();q=[]
                for candidate in range(2):
                    cf,_=rollout(model,h,x,m['source_position'],remaining,policy,observer,kind,check,first=candidate)
                    q.append(nll(model,cf,y))
                sample_audit.append(dict(remaining_steps=remaining,chosen=e,suffix_losses=q,regret=q[e]-min(q)))
                audit_seconds+=time.perf_counter()-t
            h=after
        losses.append(nll(model,h,y));routes.append(chosen)
        if audit:regrets.extend(v['regret'] for v in sample_audit)
        if file:file.write(json.dumps(dict(**m,target_id=y,loss=losses[-1],route=chosen,audit=sample_audit))+'\n')
    elapsed=time.perf_counter()-started
    return np.asarray(losses),dict(routes=routes,mean_suffix_regret=float(np.mean(regrets)) if regrets else None,
        elapsed_seconds=elapsed,audit_seconds=audit_seconds,feature_seconds=observer.seconds-feature_start)


def compare(pool,loss,baselines,seed):
    rows=[m for _,_,m in pool];out={}
    for name,b in baselines.items():
        gain=b-loss;out[name]=dict(mean_loss=float(b.mean()),gain=float(gain.mean()),gain_ci=paired_interval(rows,gain,seed))
    return out


def job(c,entry,path,root,check,emit):
    cfg,ds,model,bank,excluded,groups,info=load_model(path,c['device']);before=digest(model.state_dict())
    pools=make_pools(c,cfg,ds,excluded,groups,check)
    with (root/'sample-manifest.jsonl').open('w',encoding='utf-8') as f:
        for split,pool in pools.items():
            for x,y,m in pool:f.write(json.dumps(dict(**m,input_ids=x[0].tolist(),target_id=y))+'\n')
    write(root/'source.json',dict(source_directory=str(path),key=entry['key'],common_hash=info['common_hash'],
        final_hashes=info['final_hashes'],model_hash=before,config=saved_config(cfg),
        reference='Saved common pre-expert training bank; no expert-success history',
        sampling='Fresh balanced full-context answer endpoints. Prefix exclusions include source training/reference/evaluation. Generated-example groups disjoint across router splits and source analysis.'))
    observer=Observer(model,cfg,ds,bank,c['spectral_enabled']);results=[]
    for horizon in [1,c['sequential_steps']]:
        emit(f'{entry["key"]} · {horizon} steps · fixed-sequence training baseline')
        seqs,train_paths=path_losses(model,pools['train'],horizon,check)
        best=seqs[int(train_paths.mean(0).argmin())];initial=Policy(best)
        for kind in ['controls','full']:
            folder=root/f'{horizon}-step-{kind}';folder.mkdir()
            teacher=initial;val,_=execute(model,pools['validation'],horizon,teacher,observer,kind,check)
            accepted_loss=float(val.mean());history=[]
            for iteration in range(c['rounds']):
                check();emit(f'{entry["key"]} · {horizon} steps · {kind} · counterfactual round {iteration+1}',step=iteration+1,total=c['rounds'])
                # Teacher remains untouched throughout collection/fitting; no moving continuation targets.
                teacher.save(folder/f'round-{iteration+1}-teacher.pt')
                with (folder/f'round-{iteration+1}-targets.jsonl').open('w',encoding='utf-8') as f:
                    rows,a=collect(model,pools['train'],horizon,teacher,observer,kind,c,c['router_seed']+iteration,check,f)
                candidate=fit_policy(rows,a,best,c,c['router_seed']+iteration,check)
                candidate.save(folder/f'round-{iteration+1}-candidate.pt')
                v,_=execute(model,pools['validation'],horizon,candidate,observer,kind,check)
                score=float(v.mean());accept=score<accepted_loss-1e-8
                history.append(dict(round=iteration+1,validation_loss=score,previous_validation_loss=accepted_loss,accepted=accept,training_decisions=len(a)))
                if accept:teacher=candidate;accepted_loss=score
                emit(f'{entry["key"]} · {horizon} steps · {kind} · validation',step=iteration+1,total=c['rounds'],loss=score,accepted=accept)
                if max(abs(v) for v in a)<1e-8:
                    history[-1]['note']='All sampled counterfactual advantages zero; stop uninformative repeat rounds'
                    break
            teacher.save(folder/'router.pt');write(folder/'training.json',history)
            emit(f'{entry["key"]} · {horizon} steps · {kind} · held-out execution')
            # Timing pass has no counterfactuals. Audit is separate and never influences choices or acceptance.
            test,execution=execute(model,pools['test'],horizon,teacher,observer,kind,check)
            with (folder/'test-decisions.jsonl').open('w',encoding='utf-8') as f:
                audited,audit=execute(model,pools['test'],horizon,teacher,observer,kind,check,audit=True,file=f)
            if not np.allclose(test,audited,rtol=1e-6,atol=1e-6):raise RuntimeError('Audit changed executed loss')
            if kind=='controls':
                emit(f'{entry["key"]} · {horizon} steps · held-out fixed/random/oracle diagnostics')
                _,test_paths=path_losses(model,pools['test'],horizon,check)
                baselines=dict(fixed_sequence=test_paths[:,seqs.index(best)],uniform_random=test_paths.mean(1),oracle_sequence=test_paths.min(1))
                controls_loss=test
            comparisons=compare(pools['test'],test,baselines|({'simple_router':controls_loss} if kind=='full' else {}),c['router_seed'])
            counts=np.bincount(np.array(execution['routes']).ravel(),minlength=2).tolist()
            result=dict(horizon=horizon,features=kind,mean_loss=float(test.mean()),validation_loss=accepted_loss,
                selected_policy='learned' if teacher.net is not None else 'fixed baseline retained',history=history,
                fixed_sequence=list(best),comparisons=comparisons,expert_counts=counts,
                suffix_regret=audit['mean_suffix_regret'],test_samples=len(test),
                inference_seconds=execution['elapsed_seconds'],feature_seconds=execution['feature_seconds'],audit_seconds=audit['elapsed_seconds'],
                endpoints_per_second=len(test)/max(execution['elapsed_seconds'],1e-9),
                context_tokens_per_second=len(test)*cfg.sequence_length/max(execution['elapsed_seconds'],1e-9),
                feature_keys=teacher.keys)
            write(folder/'result.json',result);results.append(result)
    if digest(model.state_dict())!=before:raise RuntimeError('Frozen model changed during router experiment')
    return dict(key=entry['key'],family=entry['family'],seed=entry['seed'],strength=entry['strength'],status='completed',
        frozen_verified=True,source_directory=str(path),model_hash=before,results=results)


def saved_config(c):
    from dataclasses import asdict
    return asdict(c)


def aggregate(entries):
    groups={}
    for e in entries:
        if e['status']!='completed':continue
        for r in e['results']:
            key=f"{e['family']} / {e['strength']*100:g}% / {r['horizon']} steps / {r['features']}"
            group=groups.setdefault(key,[])
            group.append(dict(seed=e['seed'],gain=r['comparisons']['fixed_sequence']['gain'],
                interval=r['comparisons']['fixed_sequence']['gain_ci'],selected_policy=r['selected_policy']))
    return {key:dict(seeds=[r['seed'] for r in rows],mean_gain=float(np.mean([r['gain'] for r in rows])),
        positive_seeds=sum(r['gain']>1e-8 for r in rows),positive_intervals=sum(r['interval'][0]>0 for r in rows),
        learned_policies=sum(r['selected_policy']=='learned' for r in rows),replicates=len(rows)) for key,rows in groups.items()}


def report(result):
    lines=['# Frozen Expert Routing Test',f"Status: {result['status']}",
        'Frozen saved Token Lab experts, contextual encoder and readout; residual application h + expert(h). No production EMC Integrator or expert training.',
        'Fresh task-balanced answer prefixes. Router fit, validation selection and final test use disjoint generated-example groups. No target/task labels in router inputs.',
        'Simple controls and full input descriptors are compared at one step and the requested recurrent horizon. Full descriptors include novelty; spectra only when enabled.',
        'Training objective: |A| softplus(-sign(A) f), A = Q_B - Q_A. Q is final endpoint NLL after fixed-teacher continuation. Positive f chooses A. Ties have zero weight.',
        'Each round samples exploratory visited states, probes both alternatives, then freezes a newly fitted candidate. Accept only on lower validation final loss. Test never selects rounds/features.',
        'Fixed sequence chosen on router training inputs. Random baseline is exact expected loss over all equiprobable sequences. Oracle is an expensive diagnostic lower bound, never an inference policy.',
        'Timing excludes counterfactual audits; context tok/s counts input prefix tokens once, not generated tokens or recurrent exposures. This CPU-descriptor research path is not throughput-optimized.',
        '\n| Source | Steps | Features | Policy | Test NLL | Gain vs fixed | 95% paired interval | Gain vs random | Gain vs simple | Suffix regret |',
        '|---|---:|---|---|---:|---:|---|---:|---:|---:|']
    for e in result['entries']:
        if e['status']!='completed':lines.append(f"\n{e['key']}: {e['status']} — {e.get('error','')}");continue
        for r in e['results']:
            co=r['comparisons'];fixed=co['fixed_sequence'];simple=co.get('simple_router',{}).get('gain')
            lines.append(f"| {e['key']} | {r['horizon']} | {r['features']} | {r['selected_policy']} | {r['mean_loss']:.5f} | {fixed['gain']:.5f} | {fixed['gain_ci']} | {co['uniform_random']['gain']:.5f} | {simple} | {r['suffix_regret']:.5f} |")
    lines+=['\n## Replication across saved seeds','| Condition | Seeds | Mean gain vs fixed | Positive seeds | Positive intervals | Learned policies |','|---|---|---:|---:|---:|---:|']
    for key,g in aggregate(result['entries']).items():
        lines.append(f"| {key} | {g['seeds']} | {g['mean_gain']:.5f} | {g['positive_seeds']}/{g['replicates']} | {g['positive_intervals']}/{g['replicates']} | {g['learned_policies']}/{g['replicates']} |")
    lines+=['\n## Interpretation',
        'Positive gain = baseline loss minus executed router loss. Compare within the same checkpoint/horizon; different families and strengths are not paired observations.',
        'Report all seeds, both feature sets, both horizons and failures. Pointwise bootstrap intervals are exploratory, conditional on these fitted policies, and not corrected for multiple comparisons.',
        'A retained fixed policy is a null result, not learned routing. A sequential loss worse than one-step loss can indicate repeated expert application is harmful: specialists were trained for single-step application.',
        'This evaluates transfer from offline competence signals to frozen-bank execution on this synthetic task mixture. It does not establish cross-task transfer, emergent specialization, decentralized performance, or benefits on a jointly trained EMC bank.',
        'Share routing-test-analysis.json and report.md. Per-round raw counterfactual targets, test routes/audits, router weights, and sample manifests are preserved under jobs/.']
    return '\n'.join(lines)


def run_test(request,runs_root,run_id):
    c=validate(request);selected=sources(c)
    if not run_id or any(not (v.isascii() and (v.isalnum() or v in '-_')) for v in run_id):raise ValueError('Invalid run ID')
    root=Path(runs_root)/run_id;root.mkdir(parents=True,exist_ok=False);(root/'jobs').mkdir()
    start=time.perf_counter();started=datetime.now(timezone.utc).isoformat();entries=[]
    result=dict(schema_version=1,status='running',settings=c,entries=entries,planned_jobs=len(selected))
    write(root/'config.json',c);write(root/'metadata.json',dict(started_at=started,experiment_type='token_lab_routing',protocol_version=1))
    last_event={};last_heartbeat=[time.perf_counter()]
    def check():
        if (root/'cancel.requested').exists():raise InterruptedError('Routing test cancelled; completed results preserved')
        now=time.perf_counter()
        if last_event and now-last_heartbeat[0]>5:
            last_heartbeat[0]=now;event=last_event|dict(elapsed_seconds=now-start,heartbeat=True)
            write(root/'status.json',dict(status='running',**event));print(json.dumps(event),flush=True)
    def emit(phase,**kw):
        check();event=dict(type='token_lab_progress',run_id=run_id,phase=phase,elapsed_seconds=time.perf_counter()-start,
            completed_jobs=sum(e['status']=='completed' for e in entries),planned_jobs=len(selected),**kw)
        last_event.clear();last_event.update(event);last_heartbeat[0]=time.perf_counter()
        write(root/'status.json',dict(status='running',**event));print(json.dumps(event),flush=True)
        with (root/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
    def persist(status):
        result['status']=status;result['replication']=aggregate(entries)
        checkpoint_json(root/'routing-test-analysis.json',result);checkpoint_json(root/'analysis.json',dict(routing_test=result))
        checkpoint_json(root/'summary.json',dict(run_id=run_id,name=c['name'],started_at=started,status=status,
            experiment_type='token_lab_routing',completed_jobs=sum(e['status']=='completed' for e in entries),planned_jobs=len(selected),runtime_seconds=time.perf_counter()-start))
        (root/'report.md').write_text(report(result),encoding='utf-8')
    torch.set_num_threads(1);persist('running')
    try:
        for e,path in selected:
            check();entry=dict(key=e['key'],status='running');entries.append(entry);persist('running')
            folder=root/'jobs'/e['key'];folder.mkdir()
            try:
                emit(e['key']+' · loading verified saved specialists')
                entry.update(job(c,e,path,folder,check,emit))
            except InterruptedError:entry['status']='cancelled';raise
            except Exception as exc:
                entry.update(status='failed',error=str(exc))
                (folder/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            persist('running')
        status='completed' if all(e['status']=='completed' for e in entries) else 'partial'
        persist(status);write(root/'status.json',dict(status=status));return result
    except BaseException as exc:
        status='cancelled' if isinstance(exc,InterruptedError) else 'failed';persist(status)
        write(root/'status.json',dict(status=status,error=str(exc)));raise
