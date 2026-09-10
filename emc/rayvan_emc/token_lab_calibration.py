"""Controlled task exposure. No routing; uses Token Lab's expert delta semantics."""
import copy
import hashlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .capability_tasks import CAPABILITIES, CAPABILITY_GENERATOR_VERSION


def digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        a=v.detach().cpu().contiguous()
        h.update(k.encode());h.update(str((a.dtype,tuple(a.shape))).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()


def profiles(name,count):
    groups=[('arithmetic','symbolic','program_execution'),
            ('associative_recall','fuzzy_recall','selective_copying','working_memory','compression','stateful_action')]
    if name=='two_way' and count!=2:raise ValueError('Two-way profile requires exactly two experts')
    if name not in ['two_way','one_task','custom']:raise ValueError('Unknown specialist profile')
    return {str(i):{t:float(t in (groups[i] if name=='two_way' else (CAPABILITIES[i],)) ) for t in CAPABILITIES} for i in range(count)}


def distributions(c):
    weights=c.task_weights if c.specialization_profile=='custom' else profiles(c.specialization_profile,len(c.families))
    if not isinstance(weights,dict) or set(weights)!=set(map(str,range(len(c.families)))):raise ValueError('Provide a task distribution for every expert')
    out=[]
    for i in range(len(c.families)):
        w=weights[str(i)]
        if set(w)!=set(CAPABILITIES):raise ValueError('Each profile must contain every registered task')
        a=np.array([w[t] for t in CAPABILITIES],dtype=float)
        if not np.isfinite(a).all() or (a<0).any() or a.sum()<=0:raise ValueError('Task weights must be finite, nonnegative, with positive sum')
        a/=a.sum();p=(1-c.specialization_strength)/len(a)+c.specialization_strength*a
        out.append(dict(assigned=dict(zip(CAPABILITIES,a.tolist())),probabilities=dict(zip(CAPABILITIES,p.tolist()))))
    return out


def validate_calibration(c):
    if c.dataset!='capability_10' or c.topology!='independent':raise ValueError('Calibration requires capability_10 and independent probes')
    if len(c.families)<2:raise ValueError('Calibration requires at least two experts')
    if len(set(c.families))!=1 and not c.architecture_confounded:raise ValueError('Mixed calibration requires explicit architecture-confounded mode')
    if c.common_base_source not in ['pretrain','checkpoint']:raise ValueError('Choose common pretraining or a compatible Token Lab checkpoint')
    if c.common_base_source=='checkpoint' and not c.common_checkpoint and not c.specialist_checkpoint:raise ValueError('Common checkpoint path required')
    for key in ['common_pretrain_steps','calibration_samples_per_task']:
        if type(getattr(c,key))!=int or getattr(c,key)<1:raise ValueError(key+' must be positive')
    if c.train_steps<1:raise ValueError('Specialist training steps must be positive')
    if c.common_base_source=='pretrain' and c.common_pretrain_steps*c.batch_size%len(CAPABILITIES):raise ValueError('Common pretraining endpoint budget must be divisible by 10 for exact task balance')
    if not math.isfinite(c.specialization_strength) or not 0<=c.specialization_strength<=1:raise ValueError('Strength must be in [0,1]')
    if not 0<c.success_quantile<.5:raise ValueError('Success quantile must be in (0,.5)')
    if c.measurement_rate!=1 or c.evaluation_samples%len(CAPABILITIES):raise ValueError('Calibration requires sampling rate 1 and evaluation locations divisible by 10')
    if c.analysis_cap<c.evaluation_samples:raise ValueError('Calibration analysis cap must cover all evaluation locations')
    distributions(c)


class CalibrationDataset:
    """Existing generators; only answer targets with exactly S preceding tokens.

    No padding/masking change to expert semantics. Short examples are rejected,
    within the already chosen task, and counted. This conditions the population
    on context eligibility; it does NOT silently change the requested task mix.
    """
    def __init__(self,c,ds):
        self.c=c;self.tokenizer=ds.tokenizer;self.counts=ds.counts;self.suite=ds.suite
        self.excluded=set();self.used=set();self.rejections=0;self.short_rejections=0;self.overlap_rejections=0

    def sample_task(self,split,index,task,stream=0):
        rng=random.Random(self.c.seed+index*1000003+stream*10007+{'train':0,'validation':310000019,'evaluation':710000033}[split])
        for attempt in range(10000):
            generator_index=rng.randrange(2**31)
            e=self.suite.generate(task,split=split,index=generator_index)
            prompt=self.tokenizer.encode(e.prompt);target=self.tokenizer.encode(e.target)
            choices=[j for j in range(len(target)) if len(prompt)+j>=self.c.sequence_length]
            if not choices:self.rejections+=1;self.short_rejections+=1;continue
            j=rng.choice(choices);ids=prompt+target;endpoint=len(prompt)+j
            prefix=ids[endpoint-self.c.sequence_length:endpoint]
            key=hashlib.sha256(json.dumps(prefix).encode()).hexdigest()
            if split!='train' and (key in self.excluded or key in self.used):self.rejections+=1;self.overlap_rejections+=1;continue
            if split!='train':self.used.add(key)
            return torch.tensor(prefix,dtype=torch.long)[None],ids[endpoint],dict(
                sample_id=f'{split}-{task}-{generator_index}-{j}',analysis_group=f'{split}-{task}-{generator_index}',
                prefix_sha256=key,task_id=task,position=len(prefix)-1,source_position=endpoint-1,
                context_start=endpoint-len(prefix),sequence_length=len(prefix),dataset='capability_10',split=split,
                target_region='answer',answer_position=j)
        raise ValueError(f'Cannot obtain unique full-context answer probe for {task}. Short-context collisions may require a longer prefix; too few full-length examples require a shorter prefix. Reduce budget if exhausted. No silent padding or budget reduction.')

    def sample(self,split,index):return self.sample_task(split,index,CAPABILITIES[index%len(CAPABILITIES)],stream=61)


def load_compatible(path,c,model,tokenizer):
    saved=torch.load(path,map_location=c.device,weights_only=True)
    cfg=saved['config']
    for k in ['latent_dim','hidden_dim','heads','sequence_length','dataset']:
        if cfg[k]!=getattr(c,k):raise ValueError(f'Checkpoint mismatch: {k}')
    if saved['tokenizer']!=tokenizer.to_config():raise ValueError('Checkpoint tokenizer mismatch')
    if len(set(cfg['families']))>1 or len(set(c.families))>1:
        if list(cfg['families'])!=list(c.families):raise ValueError('Mixed checkpoint requires the exact ordered population; unrelated trunks cannot be combined')
        model.load_state_dict(saved['model'],strict=True);return saved
    if cfg['families'][0]!=c.families[0]:raise ValueError('Common expert family mismatch')
    state={k:v for k,v in saved['model'].items() if not k.startswith('experts.')}
    first={k[len('experts.0.'):]:v for k,v in saved['model'].items() if k.startswith('experts.0.')}
    for i in range(len(c.families)):state.update({f'experts.{i}.{k}':v for k,v in first.items()})
    model.load_state_dict(state,strict=True)
    return saved


def train_specialists(c,model,ds,root,excluded,check,emit):
    from .token_lab import write
    ds.excluded=excluded
    mixed=len(set(c.families))>1
    if c.specialist_checkpoint:
        source=Path(c.specialist_checkpoint)
        saved=torch.load(source/'model.pt',map_location=c.device,weights_only=True)
        info=json.loads((source/'specialization.json').read_text())
        old=saved['config']
        # A rerun may change analysis controls, not the meaning of saved specialists.
        for k in ['families','latent_dim','hidden_dim','heads','sequence_length','dataset','seed','specialization_strength','specialization_profile','task_weights','train_steps','batch_size','learning_rate','common_pretrain_steps','common_base_source','identical_stream']:
            if json.dumps(old[k],sort_keys=True)!=json.dumps(asdict(c)[k],sort_keys=True):raise ValueError('Specialist checkpoint mismatch: '+k)
        if saved['tokenizer']!=ds.tokenizer.to_config():raise ValueError('Tokenizer mismatch')
        model.load_state_dict(saved['model'],strict=True);excluded.update(saved['excluded_prefixes'])
        if [digest(e.state_dict()) for e in model.experts]!=info['final_hashes']:raise ValueError('Specialist checkpoint hash mismatch')
        info['reused_from']=str(source)
        for i,expert in enumerate(model.experts):torch.save(dict(state=expert.state_dict(),common_hash=info['common_hash'],initial_hash=info['initial_hashes'][i],final_hash=info['final_hashes'][i],budget=info['budgets'][i],distribution=info['distributions'][i]),root/f'specialist-{i+1}.pt')
        write(root/'specialization.json',info)
        return info,0,0
    model.train()
    if c.common_base_source=='checkpoint':
        saved=load_compatible(c.common_checkpoint,c,model,ds.tokenizer)
        if not saved.get('calibration_common_base'):raise ValueError('Use a saved calibration common-base.pt; arbitrary checkpoints lack verified balanced answer pretraining provenance')
        if saved['config']['common_pretrain_steps']!=c.common_pretrain_steps or saved['config']['batch_size']!=c.batch_size:raise ValueError('Common checkpoint pretraining budget differs; restore its common steps and batch size')
        excluded.update(saved['excluded_prefixes'])
    else:
        # One expert and shared machinery only; unused clones receive no updates.
        params=[p for n,p in model.named_parameters() if not n.startswith('experts.') or n.startswith('experts.0.') or mixed]
        optimizer=torch.optim.AdamW(params,lr=c.learning_rate,weight_decay=0)
        for step in range(c.common_pretrain_steps):
            check();optimizer.zero_grad(set_to_none=True);total=0.
            for b in range(c.batch_size):
                index=step*c.batch_size+b;x,y,m=ds.sample_task('train',index,CAPABILITIES[index%10],stream=11);excluded.add(m['prefix_sha256'])
                h=model.embed(x.to(c.device));active=list(model.experts) if mixed else [model.experts[0]]
                loss=sum(F.cross_entropy(model.logits(h+expert(h)),torch.tensor([y],device=c.device)) for expert in active)/(c.batch_size*len(active))
                loss.backward();total+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
            if step%10==0:emit('common pretraining',step=step+1,total=c.common_pretrain_steps,loss=total)
    common=copy.deepcopy(model.experts[0].state_dict())
    if not mixed:
        for expert in model.experts:expert.load_state_dict(common,strict=True)
    initial=[digest(e.state_dict()) for e in model.experts]
    if not mixed and len(set(initial))!=1:raise RuntimeError('Clone identity failed')
    torch.save(dict(config=asdict(c),model=model.state_dict(),tokenizer=ds.tokenizer.to_config(),excluded_prefixes=sorted(excluded),calibration_common_base=True),root/'common-base.pt')
    shared={k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith('experts.')}
    common_hash=digest(model.state_dict());dist=distributions(c);budgets=[];changed=[]
    for i,expert in enumerate(model.experts):
        check();model.eval().requires_grad_(False);expert.train().requires_grad_(True)
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        optimizer=torch.optim.AdamW(expert.parameters(),lr=c.learning_rate,weight_decay=0)
        stream=0 if c.specialization_strength==0 and c.identical_stream else i+1
        sampler=random.Random(c.seed+stream*4099+90001);counts=dict.fromkeys(CAPABILITIES,0)
        for step in range(c.train_steps):
            check();optimizer.zero_grad(set_to_none=True);total=0.
            # Same random dropout schedule for exact-stream control, if family uses it.
            torch.manual_seed(c.seed+stream*997+step)
            for b in range(c.batch_size):
                task=sampler.choices(CAPABILITIES,weights=list(dist[i]['probabilities'].values()))[0];counts[task]+=1
                x,y,m=ds.sample_task('train',step*c.batch_size+b,task,stream=100+stream);excluded.add(m['prefix_sha256'])
                with torch.no_grad():h=model.embed(x.to(c.device))
                loss=F.cross_entropy(model.logits(h+expert(h)),torch.tensor([y],device=c.device))/c.batch_size
                loss.backward();total+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(expert.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            if step%10==0:emit('specialist training',expert_id=f'{i+1}:{c.families[i]}',step=step+1,total=c.train_steps,loss=total)
        names=[k for k,v in model.state_dict().items() if not torch.equal(v,before[k])]
        if any(not k.startswith(f'experts.{i}.') for k in names):raise RuntimeError('Unexpected shared/other-expert mutation')
        if not names:raise RuntimeError('Specialist parameters did not update')
        changed.append(names)
        n=c.train_steps*c.batch_size
        budgets.append(dict(expert_id=f'{i+1}:{c.families[i]}',steps=c.train_steps,samples=n,target_tokens=n,context_tokens=n*c.sequence_length,task_counts=counts,
            realized_probabilities={t:counts[t]/n for t in CAPABILITIES},distribution_L1=sum(abs(counts[t]/n-dist[i]['probabilities'][t]) for t in CAPABILITIES)))
    if digest(shared)!=digest({k:v for k,v in model.state_dict().items() if not k.startswith('experts.')}):raise RuntimeError('Shared parameters changed')
    finals=[digest(e.state_dict()) for e in model.experts]
    info=dict(schema_version=1,generator_version=CAPABILITY_GENERATOR_VERSION,common_source=c.common_base_source,common_hash=common_hash,
        initial_hashes=initial,architecture_confounded=mixed,clone_identity_verified=not mixed,final_hashes=finals,shared_hash=digest(shared),shared_unchanged=True,
        frozen_names=list(shared),changed_names=changed,distributions=dist,strength=c.specialization_strength,budgets=budgets,
        expert_families=list(c.families),expert_parameter_counts=[sum(p.numel() for p in e.parameters()) for e in model.experts],
        common_pretraining_steps=c.common_pretrain_steps,common_pretraining_targets=c.common_pretrain_steps*c.batch_size,
        common_pretraining_expert_applications=c.common_pretrain_steps*c.batch_size*(len(c.families) if mixed else 1),
        seed=c.seed,trainable_names=[[n for n,_ in model.named_parameters() if n.startswith(f'experts.{i}.')] for i in range(len(c.families))],
        optimizer=dict(name='AdamW',learning_rate=c.learning_rate,weight_decay=0,betas=[.9,.999],epsilon=1e-8,clip_norm=1,precision='float32',schedule='constant'),
        exact_stream_control=c.specialization_strength==0 and c.identical_stream,exact_control_final_identity=len(set(finals))==1,
        answer_protocol='Fixed-length causal prefix, one uniformly sampled eligible answer character; reject short examples within chosen task. Same path for all stages.',eligibility_rejections=ds.rejections)
    for i,expert in enumerate(model.experts):torch.save(dict(state=expert.state_dict(),common_hash=common_hash,initial_hash=initial[i],final_hash=finals[i],budget=budgets[i],distribution=dist[i]),root/f'specialist-{i+1}.pt')
    write(root/'specialization.json',info)
    n=c.train_steps*c.batch_size*len(c.families)+(c.common_pretrain_steps*c.batch_size*(len(c.families) if mixed else 1) if c.common_base_source=='pretrain' else 0)
    return info,n,n*c.sequence_length


def sanity_check(c,model,ds,excluded,check,emit):
    """Separate held-out panel; unavailable to blind predictor and feature ranking."""
    from .token_lab_features import outcomes
    result={f'{i+1}:{c.families[i]}':{} for i in range(len(c.families))}
    for task in CAPABILITIES:
        records=[[] for _ in c.families]
        for j in range(c.calibration_samples_per_task):
            check();x,y,m=ds.sample_task('evaluation',j,task,stream=301)
            with torch.no_grad():
                h=model.embed(x.to(c.device));target=torch.tensor([y],device=c.device)
                base=float(F.cross_entropy(model.logits(h),target));losses=[float(F.cross_entropy(model.logits(h+e(h)),target)) for e in model.experts]
            for r,v in zip(records,outcomes([base]*len(losses),losses,True,c.tie_epsilon)):r.append(v)
            excluded.add(m['prefix_sha256'])
        for i,rs in enumerate(records):
            result[f'{i+1}:{c.families[i]}'][task]=dict(n=len(rs),mean_loss=float(np.mean([r['expert_loss'] for r in rs])),
                mean_improvement=float(np.mean([r['improvement'] for r in rs])),median_improvement=float(np.median([r['improvement'] for r in rs])),
                win_rate=float(np.mean([r['is_best'] for r in rs])),improvements=[r['improvement'] for r in rs])
        emit('held-out specialization sanity check',task=task)
    return result
