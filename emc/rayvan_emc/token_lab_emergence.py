"""Task-agnostic expert development using the existing competence learner.

Single-step residual application. Shared representation frozen. Equal numbers of
expert/example training tickets; paired random controls shuffle ticket identities.
"""
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from .token_lab import LabConfig, Dataset, ProbeModel, write
from .token_lab_calibration import CalibrationDataset, digest, load_compatible
from .token_lab_routing import Observer, Policy, fit_policy
from .token_lab_sweep import load, checkpoint_json
from . import token_lab_features as mf
from .capability_tasks import CAPABILITIES

DEFAULTS=dict(experiment_mode='emergent_specialization',name='Emergent Specialization',
    source_mode='sweep',source_sweep='',common_checkpoint='',family='gpt',expert_count=4,
    seeds=[42,43,44],asymmetry_type='weight_perturbation',asymmetry_strengths=[0.,.001,.01,.05],
    feedback_strengths=[0.,.25,.5],design='narrow',include_oracle=False,
    train_steps=250,batch_size=4,expert_learning_rate=.001,exploration=.2,
    probe_interval=50,probe_samples=256,validation_samples=200,evaluation_samples=300,
    diagnostic_samples=400,reference_size=128,fit_epochs=80,learning_rate=.001,
    checkpoints=[0.,.1,.25,.5,.75,1.],descriptor_set='full',spectral_enabled=False,
    device='cpu',tie_epsilon=.001,success_quantile=.2)


def condition_plan(c):
    aa=sorted(set(c['asymmetry_strengths'])-{0.});bb=sorted(set(c['feedback_strengths'])-{0.})
    def item(label,a,b):return dict(id=f'{label}-a{a:g}-b{b:g}',condition=label,asymmetry=a,feedback=b)
    out=[item('A',0.,0.)]+[item('B',a,0.) for a in aa]+[item('C',0.,b) for b in bb]
    pairs=list(itertools.product(aa,bb))
    if c['design']=='narrow':pairs=[(a,b) for a,b in pairs if b==max(bb) or a==aa[len(aa)//2]]
    for a,b in pairs:
        out.extend([item('D',a,b),item('E',a,b)])
        if c['include_oracle']:out.append(item('F',a,b))
    return out


def base_path(c,seed):
    if c['source_mode']=='checkpoint':return Path(c['common_checkpoint']).resolve()
    root=Path(c['source_sweep']);sweep=load(root/'sweep-analysis.json')
    e=next((e for e in sweep['entries'] if e.get('status')=='completed' and e['family']==c['family'] and e['seed']==seed and e['strength']==1),None)
    if e is None:raise ValueError(f'No completed {c["family"]} seed {seed} common base in selected sweep')
    local=root/'jobs'/e['key']/'common-base.pt'
    return (local if local.is_file() else Path(e['source_directory'])/'common-base.pt').resolve()


def validate(request):
    c={k:request.get(k,copy.deepcopy(v)) for k,v in DEFAULTS.items()}
    if c['source_mode'] not in ['sweep','checkpoint']:raise ValueError('Choose sweep common bases or one common checkpoint')
    if not c['source_sweep'] and c['source_mode']=='sweep':raise ValueError('Select a saved validation sweep')
    if not c['common_checkpoint'] and c['source_mode']=='checkpoint':raise ValueError('Select a calibration common-base.pt, not specialized model.pt')
    if c['family'] not in ['gpt','ssm','recurrent','delta'] or c['expert_count'] not in [2,4,8]:raise ValueError('Use 2, 4 or 8 homogeneous experts of a supported family')
    if c['asymmetry_type']!='weight_perturbation':raise ValueError('Primary experiment supports weight perturbation only')
    if c['design'] not in ['narrow','factorial']:raise ValueError('Unknown design')
    if c['descriptor_set'] not in ['controls','full']:raise ValueError('Unknown descriptor set')
    if not isinstance(c['seeds'],list) or not 1<=len(c['seeds'])<=20 or len(set(c['seeds']))!=len(c['seeds']) or any(type(s) is not int or not 0<=s<2**30 for s in c['seeds']):raise ValueError('Provide 1–20 distinct nonnegative seeds')
    for key,hi in [('asymmetry_strengths',.5),('feedback_strengths',1.)]:
        v=c[key]
        if not isinstance(v,list) or not 2<=len(v)<=8 or any(type(a) not in [int,float] or not math.isfinite(a) or not 0<=a<=hi for a in v) or 0 not in v or not any(a>0 for a in v):raise ValueError(f'{key}: include zero and positive values up to {hi}')
    for k,lo,hi in [('train_steps',1,100000),('batch_size',1,64),('probe_interval',1,10000),('probe_samples',20,4096),('validation_samples',20,5000),('evaluation_samples',20,10000),('diagnostic_samples',100,5000),('reference_size',10,1024),('fit_epochs',1,500)]:
        if type(c[k]) is not int or not lo<=c[k]<=hi:raise ValueError(f'{k} must be an integer in [{lo},{hi}]')
    if c['train_steps']*c['batch_size']%10:raise ValueError('Training endpoint budget must be divisible by 10')
    for k in ['evaluation_samples','validation_samples','diagnostic_samples']:
        if c[k]%10:raise ValueError(k+' must be divisible by 10')
    if not 0<c['exploration']<=1:raise ValueError('Mandatory exploration must be in (0,1]')
    for k in ['expert_learning_rate','learning_rate','tie_epsilon']:
        if not math.isfinite(c[k]) or c[k]<=0:raise ValueError('Invalid '+k)
    if not 0<c['success_quantile']<.5:raise ValueError('Success quantile must be in (0,.5)')
    if not isinstance(c['checkpoints'],list) or not 2<=len(c['checkpoints'])<=10 or any(type(v) not in [int,float] or not math.isfinite(v) or not 0<=v<=1 for v in c['checkpoints']) or not {0,1}<=set(c['checkpoints']):raise ValueError('Checkpoint fractions must include 0 and 1')
    if c['device'] not in ['cpu','cuda'] or (c['device']=='cuda' and not torch.cuda.is_available()):raise ValueError('Device unavailable')
    for k in ['include_oracle','spectral_enabled']:
        if type(c[k]) is not bool:raise ValueError(k+' must be boolean')
    for s in c['seeds']:
        p=base_path(c,s)
        if not p.is_file():raise ValueError(f'Missing common base: {p}')
    return c


def checkpoint_steps(c):return sorted({round(c['train_steps']*v) for v in c['checkpoints']})


def cost(c):
    n=len(condition_plan(c))*len(c['seeds']);E=c['expert_count']
    refresh=len(set(range(0,c['train_steps']+1,c['probe_interval']))|set(checkpoint_steps(c)))
    return dict(jobs=n,training_applications=n*c['train_steps']*c['batch_size']*E,
        checkpoint_panels=len(checkpoint_steps(c)),refreshes_per_job=refresh,
        counterfactual_applications_without_optional_oracle=n*E*(refresh*(c['probe_samples']+c['validation_samples'])+len(checkpoint_steps(c))*(c['evaluation_samples']+c['diagnostic_samples'])),
        maximum_pair_fits=n*refresh*E*(E-1)//2,
        diagnostic_probes=n*len(checkpoint_steps(c))*E,
        note='Counts exclude common encoder, optional online oracle and MLP fitting. No reliable FLOP counter; wall-time estimate becomes available after first job.')


def shared_state(model):return {k:v for k,v in model.state_dict().items() if not k.startswith('experts.')}


def initialize(c,seed):
    path=base_path(c,seed);saved=torch.load(path,map_location='cpu',weights_only=True)
    if not saved.get('calibration_common_base'):raise ValueError('Expected verified general calibration common-base.pt')
    if c['source_mode']=='sweep':
        source=next(v for v in load(Path(c['source_sweep'])/'sweep-analysis.json')['entries'] if v['family']==c['family'] and v['seed']==seed and v['strength']==1)
        if digest(saved['model'])!=source['controls']['common_hash']:raise ValueError('Sweep/common-base hash mismatch')
    cfg=LabConfig(**(saved['config']|dict(experiment_mode='standard',topology='independent',device=c['device'],families=[c['family']]*c['expert_count'],seed=saved['config']['seed']))).validate()
    if saved['config']['families'][0]!=c['family'] or cfg.dataset!='capability_10':raise ValueError('Common base family/dataset mismatch')
    ds=Dataset(cfg);model=ProbeModel(cfg,ds.tokenizer.vocab_size).to(c['device'])
    load_compatible(path,cfg,model,ds.tokenizer)
    hashes=[digest(e.state_dict()) for e in model.experts]
    if len(set(hashes))!=1:raise RuntimeError('Common clone identity failed')
    model.eval().requires_grad_(False)
    return cfg,ds,model,set(saved['excluded_prefixes']),dict(path=str(path),common_hash=digest(saved['model']),shared_hash=digest(shared_state(model)),clone_hashes=hashes,clones_identical=True,
        frozen_parameter_names=[n for n,_ in model.named_parameters() if not n.startswith('experts.')],trainable_parameter_names=[n for n,_ in model.named_parameters() if n.startswith('experts.')])


def perturb(model,strength,seed):
    """Each expert's global parameter perturbation has exact L2 = a*||theta||."""
    initial=[digest(e.state_dict()) for e in model.experts]
    if len(set(initial))!=1:raise ValueError('Perturbation requires identical clones')
    details=[]
    for i,e in enumerate(model.experts):
        gen=torch.Generator().manual_seed(seed+104729*(i+1));params=list(e.parameters())
        norm=math.sqrt(sum(float(p.detach().double().square().sum()) for p in params))
        noise=[torch.randn(p.shape,generator=gen,dtype=torch.float32) for p in params]
        size=math.sqrt(sum(float(v.double().square().sum()) for v in noise));before=[p.detach().clone() for p in params]
        with torch.no_grad():
            for p,v in zip(params,noise):p.add_(v.to(p.device),alpha=strength*norm/max(size,1e-20))
        realized=math.sqrt(sum(float((p.detach()-b).double().square().sum()) for p,b in zip(params,before)))
        details.append(dict(expert=i,perturbation_seed=seed+104729*(i+1),base_norm=norm,delta_norm=realized,relative_delta_norm=realized/max(norm,1e-20),initial_hash=initial[i],perturbed_hash=digest(e.state_dict())))
    if strength==0 and len({d['perturbed_hash'] for d in details})!=1:raise RuntimeError('Zero perturbation changed identity')
    return dict(mechanism='independent Gaussian direction normalized to global expert parameter L2',strength=strength,experts=details,pairwise=parameter_distances(model))


def parameter_distances(model):
    out=[]
    for i,j in itertools.combinations(range(len(model.experts)),2):
        a=torch.cat([p.detach().flatten().cpu() for p in model.experts[i].parameters()]);b=torch.cat([p.detach().flatten().cpu() for p in model.experts[j].parameters()])
        out.append(dict(a=i,b=j,l2=float((a-b).norm()),relative_l2=float((a-b).norm()/((a.norm()+b.norm())/2).clamp_min(1e-8))))
    return out


class PopulationRouter:
    """Same pairwise Policy learner. >2 experts use pairwise wins, then preference to break ties."""
    def __init__(self,count,pairs=None,fixed=None,neutral=()):self.count=count;self.pairs=pairs or {};self.fixed=fixed;self.neutral=set(neutral)
    def scores(self,row):
        if not self.pairs:
            s=np.ones(self.count)/2
            if self.fixed is not None:s[self.fixed]=1.
            return s,0.
        total=np.zeros(self.count);wins=np.zeros(self.count);max_logit=0.
        for (i,j),p in self.pairs.items():
            if (i,j) in self.neutral:prob=.5
            elif p.net is None:prob=1. if p.sequence[0]==0 else 0.
            else:
                x=np.array([row.get(k) if row.get(k) is not None else np.nan for k in p.keys],dtype=np.float32)
                x=np.where(np.isfinite(x),x,p.median);z=torch.from_numpy(np.clip((x-p.median)/p.scale,-20,20)).float()[None]
                with torch.no_grad():logit=float(p.net(z)[0,0])
                max_logit=max(max_logit,abs(logit));prob=float(torch.sigmoid(torch.tensor(logit)))
                # Preserve the original sign decision even when float32 sigmoid
                # rounds a tiny nonzero logit to exactly one half.
                if prob==.5 and logit!=0:prob=float(np.nextafter(.5,1. if logit>0 else 0.))
            total[i]+=prob;total[j]+=1-prob
            vote=.5 if prob==.5 else float(prob>.5);wins[i]+=vote;wins[j]+=1-vote
        if self.count==2:return total,max_logit
        # A half-win exceeds the entire soft tiebreak range. A consistent pairwise
        # winner cannot lose to another expert merely because its margins are small.
        return (wins+.25*total/(self.count-1))/(self.count-1+.25),max_logit
    def choose(self,row):
        s,_=self.scores(row)
        # Preserve the validated two-expert tie convention (B when logit == 0).
        return int(s[1]>=s[0]) if self.count==2 else int(s.argmax())
    def save(self,path):
        path.mkdir(exist_ok=True);write(path/'router.json',dict(count=self.count,fixed=self.fixed,pairs=[list(p) for p in self.pairs],neutral=[list(p) for p in self.neutral],aggregation='pairwise win count, then mean preference; two-expert original sign rule'))
        for (i,j),p in self.pairs.items():p.save(path/f'pair-{i}-{j}.pt')
    @classmethod
    def load(cls,path):
        path=Path(path);v=load(path/'router.json')
        return cls(v['count'],{(i,j):Policy.load(path/f'pair-{i}-{j}.pt') for i,j in v['pairs']},v['fixed'],[tuple(p) for p in v.get('neutral',[])])


def fit_router(rows,losses,c,seed,check):
    E=losses.shape[1];pairs={};neutral=[]
    if np.max(np.ptp(losses,axis=1))<1e-8:return PopulationRouter(E)
    for i,j in itertools.combinations(range(E),2):
        a=losses[:,j]-losses[:,i];fallback=0 if a.mean()>0 else 1
        if np.max(abs(a))<1e-8:neutral.append((i,j))
        pairs[i,j]=fit_policy(rows,a,[fallback],c,seed+i*E+j,check)
    return PopulationRouter(E,pairs,neutral=neutral)


def assignment_tickets(rows,router,feedback,exploration,rng,count,oracle=None):
    """E tickets per input; uniform rotation at beta=0; redirected Bernoulli tickets."""
    assignments=np.tile(np.arange(count),len(rows));redirected=0;explored=0
    for b,row in enumerate(rows):
        scores,_=router.scores(row) if oracle is None else (None,None)
        for slot in range(count):
            if rng.random()>=feedback:continue
            redirected+=1
            if rng.random()<exploration:
                e=rng.randrange(count);explored+=1
            elif oracle is not None:
                tied=np.flatnonzero(np.isclose(oracle[b],oracle[b].min(),rtol=0,atol=1e-8));e=int(rng.choice(tied.tolist()))
            else:
                tied=np.flatnonzero(np.isclose(scores,scores.max(),rtol=0,atol=1e-8));e=int(rng.choice(tied.tolist()))
            assignments[b*count+slot]=e
    return assignments,dict(redirected=redirected,explored=explored)


def randomize_tickets(assignments,rng):
    out=list(map(int,assignments));rng.shuffle(out);return np.array(out,dtype=int)


def make_data(c,seed,cfg,ds,model,excluded,check):
    data=CalibrationDataset(replace(cfg,seed=seed+500000003),ds);used=set(excluded);groups=set();pools={}
    sizes=dict(reference=c['reference_size'],train=c['train_steps']*c['batch_size'],probe=c['probe_samples'],validation=c['validation_samples'],evaluation=c['evaluation_samples'],diagnostic=c['diagnostic_samples'])
    cache_bytes=sum(sizes.values())*cfg.sequence_length*(cfg.latent_dim*4+8)
    if cache_bytes>2*1024**3:raise ValueError('Matched-state cache would exceed 2 GiB; reduce training/panel budget or checkpoint size')
    for number,(name,n) in enumerate(sizes.items()):
        pool=[];tasks=[CAPABILITIES[i%10] for i in range(n)]
        random.Random(seed+number).shuffle(tasks)
        for i,task in enumerate(tasks):
            check()
            for attempt in range(10000):
                x,y,m=data.sample_task('train' if name in ['reference','train','probe'] else 'evaluation',i+attempt*100003,task,stream=2000+number)
                if m['prefix_sha256'] in used or m['analysis_group'] in groups:continue
                used.add(m['prefix_sha256']);groups.add(m['analysis_group']);m['development_split']=name
                with torch.no_grad():h=model.embed(x.to(c['device'])).cpu()
                pool.append(dict(x=x,y=y,meta=m,h=h));break
            else:raise ValueError('Cannot construct disjoint balanced development panels; reduce budget or use a longer common-base context')
        pools[name]=pool
    values=torch.tensor([[mf.core(v['h'][0,-cfg.window:]).get(k) for k in mf.NOVEL_DESCRIPTOR] for v in pools['reference']])
    mean=values.mean(0);std=values.std(0,unbiased=False).clamp_min(.001)
    bank=dict(hidden=torch.stack([F.normalize(v['h'][0,-1],dim=0) for v in pools['reference']]),descriptor=(values-mean)/std,mean=mean,std=std)
    observer=Observer(model,cfg,ds,bank,c['spectral_enabled'])
    for name,pool in pools.items():
        if name=='reference':continue
        for v in pool:
            check();v['features']=observer.features(v['h'].to(c['device']),v['x'],v['meta']['source_position'],1,1,c['descriptor_set'])
    return pools,bank


@torch.no_grad()
def measure(model,pool,check):
    model.eval();device=next(model.parameters()).device;losses=[];baselines=[];disagreement=[]
    for v in pool:
        check();h=v['h'].to(device);y=torch.tensor([v['y']],device=device)
        baselines.append(float(F.cross_entropy(model.logits(h),y)));ls=[];deltas=[]
        for expert in model.experts:
            d=expert(h);ls.append(float(F.cross_entropy(model.logits(h+d),y)));deltas.append(d[0,-1])
        losses.append(ls);ds=torch.stack(deltas)
        disagreement.append(float(torch.pdist(ds).square().mean().sqrt()))
    return np.array(losses),np.array(baselines),np.array(disagreement)


def train_batch(model,batch,tickets,optimizers,seed,check):
    """Exactly E*B physical expert applications, including repeated input tickets."""
    E=len(model.experts);B=len(batch);device=next(model.parameters()).device;updates=np.zeros(E,dtype=int);counts=np.bincount(tickets,minlength=E)
    locations=np.repeat(np.arange(B),E)
    model.eval().requires_grad_(False)
    for i,expert in enumerate(model.experts):
        check();indices=locations[tickets==i]
        if not len(indices):continue
        expert.train().requires_grad_(True);torch.manual_seed(seed)
        h=torch.cat([batch[j]['h'] for j in indices]).to(device);y=torch.tensor([batch[j]['y'] for j in indices],device=device)
        optimizers[i].zero_grad(set_to_none=True)
        loss=F.cross_entropy(model.logits(h+expert(h)),y,reduction='sum')/B
        loss.backward();torch.nn.utils.clip_grad_norm_(expert.parameters(),1.,error_if_nonfinite=True);optimizers[i].step();updates[i]=1
        expert.eval().requires_grad_(False)
    return counts,updates


def train_condition(c,seed,condition,base,pools,bank,cfg,ds,root,check,emit,replay=None):
    from .token_lab_emergence_analysis import checkpoint_analysis
    model=copy.deepcopy(base);frozen=digest(shared_state(model));initial=perturb(model,condition['asymmetry'],seed)
    write(root/'initialization.json',initial)
    optimizers=[torch.optim.AdamW(e.parameters(),lr=c['expert_learning_rate'],weight_decay=0) for e in model.experts]
    E=c['expert_count'];router=PopulationRouter(E);exposure=np.zeros(E,dtype=int);updates=np.zeros(E,dtype=int);probes=np.zeros(E,dtype=int)
    counter=dict(training_applications=0,training_context_tokens=0,unique_training_endpoints=0,probe_applications=0,probe_context_tokens=0,oracle_training_applications=0,router_fit_seconds=0.,probe_seconds=0.,training_seconds=0.,diagnostic_seconds=0.,redirected=0,explored=0)
    checkpoints=[];plans=[];previous=None;start=time.perf_counter();rng=random.Random(seed+97001);shuffle=random.Random(seed+198001)
    steps=checkpoint_steps(c);refresh=set(range(0,c['train_steps']+1,c['probe_interval']))|set(steps)
    task_exposure=[dict.fromkeys(CAPABILITIES,0) for _ in range(E)]
    with (root/'training-assignments.jsonl').open('w',encoding='utf-8') as allocation_file:
        for step in range(c['train_steps']+1):
            check()
            if step in refresh:
                emit(condition['id']+' · counterfactual refresh',step=step,total=c['train_steps'])
                t=time.perf_counter();trainL,_,_=measure(model,pools['probe'],check);valL,_,_=measure(model,pools['validation'],check)
                amount=len(pools['probe'])+len(pools['validation']);probes+=amount;counter['probe_applications']+=E*amount;counter['probe_seconds']+=time.perf_counter()-t
                counter['probe_context_tokens']+=E*sum(v['x'].numel() for name in ['probe','validation'] for v in pools[name])
                fixed=int(trainL.mean(0).argmin());rows=[v['features'] for v in pools['probe']]
                refreshdir=root/f'refresh-{step}';refreshdir.mkdir()
                with (refreshdir/'counterfactuals.jsonl').open('w',encoding='utf-8') as f:
                    for v,ls in zip(pools['probe'],trainL):f.write(json.dumps(dict(**v['meta'],features=v['features'],losses=ls.tolist()))+'\n')
                if condition['condition']!='F':
                    t=time.perf_counter();candidate=fit_router(rows,trainL,c,seed+step,check);counter['router_fit_seconds']+=time.perf_counter()-t
                    options=[router,candidate,PopulationRouter(E,fixed=fixed)]
                    scores=[float(np.mean([ls[p.choose(v['features'])] for v,ls in zip(pools['validation'],valL)])) for p in options]
                    winner=int(np.argmin(scores));router=options[winner]
                    write(refreshdir/'selection.json',dict(validation_losses=scores,selected=['previous','candidate','training-fixed'][winner],test_used=False))
                    router.save(refreshdir/'accepted-router')
                if step in steps:
                    emit(condition['id']+' · checkpoint evaluation',step=step,total=c['train_steps'])
                    t=time.perf_counter();L,baseline,out=measure(model,pools['evaluation'],check);D,db,_=measure(model,pools['diagnostic'],check)
                    amount=len(L)+len(D);probes+=amount;counter['probe_applications']+=amount*E;counter['probe_seconds']+=time.perf_counter()-t
                    counter['probe_context_tokens']+=E*sum(v['x'].numel() for name in ['evaluation','diagnostic'] for v in pools[name])
                    folder=root/f'checkpoint-{step}';folder.mkdir()
                    t=time.perf_counter();metrics,previous=checkpoint_analysis(c,seed,step,pools,L,baseline,out,D,db,router,fixed,previous,folder,check,oracle_policy=condition['condition']=='F')
                    counter['diagnostic_seconds']+=time.perf_counter()-t
                    if digest(shared_state(model))!=frozen:raise RuntimeError('Frozen shared parameters changed')
                    metrics.update(step=step,condition=condition['condition'],asymmetry=condition['asymmetry'],feedback=condition['feedback'],exposure=exposure.tolist(),optimizer_updates=updates.tolist(),probes_per_expert=probes.tolist(),task_exposure=copy.deepcopy(task_exposure),parameter_distances=parameter_distances(model),expert_hashes=[digest(e.state_dict()) for e in model.experts],cost=counter.copy(),wall_seconds=time.perf_counter()-start,shared_unchanged=True)
                    if step and exposure.min()<.05*exposure.sum()/E:metrics['warnings'].append('Training starvation: an expert received less than 5% of equal-share exposure')
                    if step and exposure.max()>.95*exposure.sum():metrics['warnings'].append('Training concentration above 95%')
                    if c['probe_samples']<100:metrics['warnings'].append('Fewer than 100 current training probe states per expert; competence evidence may be weak')
                    if condition['condition']=='A' and len({digest(e.state_dict()) for e in model.experts})!=1:raise RuntimeError('Identical-exposure control diverged unexpectedly')
                    torch.save(dict(config=asdict(cfg),model=model.state_dict(),optimizer_states=[o.state_dict() for o in optimizers],condition=condition,step=step),folder/'model.pt');router.save(folder/'router')
                    for i,e in enumerate(model.experts):torch.save(e.state_dict(),folder/f'expert-{i}.pt')
                    write(folder/'metrics.json',metrics);checkpoints.append(metrics)
                    write(root/'trajectory.json',checkpoints)
                    emit(condition['id']+' · checkpoint complete',seed=seed,condition_id=condition['id'],step=step,total=c['train_steps'],loss=metrics['utility']['policy_loss'],interaction_rms=metrics['specialization']['interaction_rms'],router_gain=metrics['utility']['gain_over_fixed'],training_context_tokens=counter['training_context_tokens'],training_targets=counter['training_applications'])
            if step==c['train_steps']:break
            batch=pools['train'][step*c['batch_size']:(step+1)*c['batch_size']]
            if condition['condition']=='E':
                if replay is None:raise ValueError('Matched learned allocation required for random control')
                assigned=randomize_tickets(replay[step]['tickets'],shuffle);event=replay[step]['event'].copy()
                if sorted(assigned.tolist())!=sorted(replay[step]['tickets']):raise RuntimeError('Random control exposure budget mismatch')
            else:
                oracle=None
                if condition['condition']=='F':
                    t=time.perf_counter();oracle,_,_=measure(model,batch,check);counter['probe_seconds']+=time.perf_counter()-t;counter['oracle_training_applications']+=len(batch)*E
                assigned,event=assignment_tickets([v['features'] for v in batch],router,condition['feedback'],c['exploration'],rng,E,oracle)
            plans.append(dict(tickets=assigned.tolist(),event=event))
            allocation_file.write(json.dumps(dict(step=step,sample_ids=[v['meta']['sample_id'] for v in batch],tickets=assigned.tolist(),**event))+'\n')
            for location,e in zip(np.repeat(np.arange(len(batch)),E),assigned):task_exposure[e][batch[location]['meta']['task_id']]+=1
            t=time.perf_counter();n,u=train_batch(model,batch,assigned,optimizers,seed+step,check);counter['training_seconds']+=time.perf_counter()-t
            exposure+=n;updates+=u;counter['training_applications']+=len(assigned)
            counter['training_context_tokens']+=E*sum(v['x'].numel() for v in batch);counter['unique_training_endpoints']+=len(batch)
            counter['redirected']+=event['redirected'];counter['explored']+=event['explored']
    if sum(exposure)!=c['train_steps']*c['batch_size']*E:raise RuntimeError('Training application budget mismatch')
    return dict(**condition,seed=seed,status='completed',initialization=initial,checkpoints=checkpoints,final=checkpoints[-1],cost=counter,shared_unchanged=True),plans


def run_study(request,runs_root,run_id):
    from .token_lab_emergence_analysis import summarize,report
    c=validate(request);plan=condition_plan(c)
    if not run_id or any(not(v.isascii() and (v.isalnum() or v in '-_')) for v in run_id):raise ValueError('Invalid run ID')
    root=Path(runs_root)/run_id;root.mkdir(parents=True,exist_ok=False);(root/'jobs').mkdir();started=datetime.now(timezone.utc).isoformat();start=time.perf_counter()
    entries=[];result=dict(schema_version=1,status='running',settings=c,plan=plan,planned_jobs=len(plan)*len(c['seeds']),entries=entries,cost_estimate=cost(c))
    write(root/'config.json',c);write(root/'metadata.json',dict(experiment_type='token_lab_emergence',started_at=started,torch_version=torch.__version__,protocol_version=1))
    last=[time.perf_counter()];phase=['preparing']
    def check():
        if (root/'cancel.requested').exists():raise InterruptedError('Cancelled; completed checkpoints preserved')
        if time.perf_counter()-last[0]>5:emit(phase[0],heartbeat=True)
    def emit(name,**kw):
        phase[0]=name;last[0]=time.perf_counter();finished=[e['final']['wall_seconds'] for e in entries if e['status']=='completed'];remaining=result['planned_jobs']-len(finished)
        event=dict(estimated_remaining_seconds=float(np.mean(finished)*remaining) if finished else None,type='token_lab_progress',run_id=run_id,phase=name,elapsed_seconds=last[0]-start,completed_jobs=sum(e['status']=='completed' for e in entries),planned_jobs=result['planned_jobs'],**kw)
        write(root/'status.json',dict(status='running',**event));print(json.dumps(event),flush=True)
        if not kw.get('heartbeat'):
            with (root/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
    def persist(status):
        result['status']=status;result['summary']=summarize(entries,c)
        checkpoint_json(root/'emergence-analysis.json',result);checkpoint_json(root/'analysis.json',dict(emergence=result))
        checkpoint_json(root/'summary.json',dict(run_id=run_id,name=c['name'],status=status,started_at=started,experiment_type='token_lab_emergence',completed_jobs=sum(e['status']=='completed' for e in entries),planned_jobs=result['planned_jobs'],runtime_seconds=time.perf_counter()-start))
        (root/'report.md').write_text(report(result),encoding='utf-8')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);persist('running')
    try:
        for seed in c['seeds']:
            check();emit(f'seed {seed} · common base and matched data')
            preparation=time.perf_counter()
            try:
                cfg,ds,base,excluded,provenance=initialize(c,seed);pools,bank=make_data(c,seed,cfg,ds,base,excluded,check)
            except InterruptedError:raise
            except Exception as exc:
                entries.extend(dict(**v,seed=seed,status='failed',error='Common base/data preparation: '+str(exc)) for v in plan)
                (root/f'seed-{seed}-error.txt').write_text(traceback.format_exc(),encoding='utf-8');persist('running');continue
            provenance['preparation_seconds']=time.perf_counter()-preparation
            parent=root/f'seed-{seed}';parent.mkdir();write(parent/'common-base.json',provenance);torch.save(dict(config=asdict(cfg),model=base.state_dict(),tokenizer=ds.tokenizer.to_config(),excluded_prefixes=sorted(excluded),calibration_common_base=True),parent/'common-base.pt');torch.save(bank,parent/'reference-bank.pt')
            torch.save(pools,parent/'matched-data.pt')
            with (parent/'sample-manifest.jsonl').open('w',encoding='utf-8') as f:
                for pool in pools.values():
                    for v in pool:f.write(json.dumps(dict(**v['meta'],input_ids=v['x'][0].tolist(),target_id=v['y']))+'\n')
            replays={}
            for cond in plan:
                check();entry=dict(**cond,seed=seed,status='running');entries.append(entry);persist('running');folder=root/'jobs'/f'seed-{seed}-{cond["id"]}';folder.mkdir()
                try:
                    data,alloc=train_condition(c,seed,cond,base,pools,bank,cfg,ds,folder,check,emit,replays.get((cond['asymmetry'],cond['feedback'])))
                    entry.update(data)
                    if cond['condition']=='D':replays[(cond['asymmetry'],cond['feedback'])]=alloc
                except InterruptedError:entry['status']='cancelled';raise
                except Exception as exc:
                    entry.update(status='failed',error=str(exc));(folder/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
                persist('running')
        status='completed' if all(e['status']=='completed' for e in entries) else 'partial';persist(status);write(root/'status.json',dict(status=status));return result
    except BaseException as exc:
        status='cancelled' if isinstance(exc,InterruptedError) else 'failed';persist(status);write(root/'status.json',dict(status=status,error=str(exc)));raise
