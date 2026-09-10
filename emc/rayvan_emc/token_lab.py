"""Token Lab CLI. Observational modes and an explicitly separate frozen routing test."""
import argparse
from dataclasses import dataclass,asdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time
import torch
from torch import nn
from torch.nn import functional as F
from .model import EMCConfig
from .modules import create_emc_module
from .data import CharacterTokenizer,load_tinystories
from .capability_tasks import CapabilityTaskSuite,CapabilitySuiteConfig,CAPABILITIES
from . import token_lab_features as mf
from .token_lab_analysis import analyze,report


@dataclass
class LabConfig:
    name:str='Token Lab'
    dataset:str='tinystories'
    families:tuple=('gpt','gpt')
    topology:str='independent'
    latent_dim:int=64
    hidden_dim:int=128
    heads:int=4
    seed:int=42
    sequence_length:int=64
    train_steps:int=100
    learning_rate:float=.001
    evaluation_samples:int=256
    window:int=16
    batch_size:int=4
    measurement_rate:float=1.
    spectral_enabled:bool=True
    spectral_rate:float=1.
    deep_enabled:bool=False
    deep_rate:float=.05
    device:str='cpu'
    save_raw:bool=True
    reference_size:int=128
    analysis_cap:int=2000
    knn_k:int=4
    tie_epsilon:float=.001
    train_stories:int=1000
    validation_stories:int=500
    experiment_mode:str='standard'
    common_base_source:str='pretrain'
    common_checkpoint:str=''
    specialist_checkpoint:str=''
    common_pretrain_steps:int=1000
    specialization_strength:float=1.
    specialization_profile:str='two_way'
    task_weights:dict=None
    identical_stream:bool=True
    calibration_samples_per_task:int=50
    success_quantile:float=.2

    def validate(self):
        if self.experiment_mode not in ['standard','forced_specialization']:raise ValueError('Unknown experiment mode')
        if self.experiment_mode=='forced_specialization':
            from .token_lab_calibration import validate_calibration
            validate_calibration(self)
        if self.dataset not in ['tinystories','capability_10']:raise ValueError('Unsupported dataset')
        if self.topology not in ['independent','serial']:raise ValueError('Unsupported topology')
        if not 1<=len(self.families)<=8 or any(f not in ['gpt','ssm','recurrent','delta'] for f in self.families):raise ValueError('Choose 1–8 supported experts')
        for name in ['latent_dim','hidden_dim','heads','sequence_length','evaluation_samples','batch_size','reference_size','analysis_cap','knn_k','train_stories','validation_stories']:
            v=getattr(self,name)
            if type(v)!=int or v<1:raise ValueError(name+' must be a positive integer')
        if self.latent_dim%self.heads or self.latent_dim<4:raise ValueError('Latent dimension must be >=4 and divisible by heads')
        if self.window not in [4,8,16,32]:raise ValueError('Window must be 4, 8, 16 or 32')
        if self.sequence_length<4 or self.sequence_length>512 or self.latent_dim>512:raise ValueError('This lab supports context 4–512 and latent dimension up to 512')
        if type(self.seed)!=int or self.seed<0:raise ValueError('Seed must be a nonnegative integer')
        if 'delta' in self.families and self.sequence_length*self.heads*(self.latent_dim//self.heads)**2*4>64*2**20:raise ValueError('Delta transition exceeds 64 MiB safety limit: reduce context or latent dimension, or increase heads')
        if type(self.train_steps)!=int or self.train_steps<0:raise ValueError('train_steps must be a nonnegative integer')
        for name in ['measurement_rate','spectral_rate','deep_rate']:
            if not math.isfinite(getattr(self,name)) or not 0<=getattr(self,name)<=1:raise ValueError(name+' must be in [0,1]')
        if self.measurement_rate<=0:raise ValueError('Measurement rate must be positive')
        if not math.isfinite(self.learning_rate) or self.learning_rate<=0 or not math.isfinite(self.tie_epsilon) or self.tie_epsilon<0:raise ValueError('Invalid learning rate/tie epsilon')
        if self.device not in ['cpu','cuda']:raise ValueError('Device must be cpu or cuda')
        if self.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable in configured Python environment')
        return self


class Dataset:
    def __init__(self,c):
        self.c=c;self.suite=CapabilityTaskSuite(CapabilitySuiteConfig(seed=c.seed))
        if c.dataset=='tinystories':
            self.corpus=load_tinystories(max_train_stories=c.train_stories,max_validation_stories=c.validation_stories);self.tokenizer=self.corpus.tokenizer
            self.counts=torch.bincount(self.corpus.train_tokens,minlength=self.tokenizer.vocab_size)
        else:
            # Fixed printable vocabulary; never infer vocabulary/frequencies from held-out texts.
            self.tokenizer=CharacterTokenizer(chr(i) for i in range(9,127))
            self.counts=torch.zeros(self.tokenizer.vocab_size,dtype=torch.long)
            for i in range(1000):self.counts+=torch.bincount(torch.tensor(self.tokenizer.encode(self.suite.generate(CAPABILITIES[i%10],split='train',index=i).model_text)),minlength=self.tokenizer.vocab_size)

    def sample(self,split,index):
        c=self.c;rng=random.Random(c.seed+index*1009+(0 if split=='train' else 100000007))
        if c.dataset=='tinystories':
            tokens=self.corpus.train_tokens if split=='train' else self.corpus.validation_tokens
            blocks=(len(tokens)//(c.sequence_length+1))
            if not blocks:raise ValueError('Dataset subset too small for context')
            if split!='train' and index>=blocks:raise ValueError('Evaluation budget exceeds disjoint validation blocks; increase validation stories')
            block=rng.randrange(blocks) if split=='train' else index
            ids=tokens[block*(c.sequence_length+1):(block+1)*(c.sequence_length+1)].tolist();task='language';sample=f'{split}-block-{block}'
        else:
            task=CAPABILITIES[index%10];example=self.suite.generate(task,split=split,index=index)
            ids=self.tokenizer.encode(example.model_text);sample=f'{split}-{task}-{index}'
        endpoint=rng.randint(2,len(ids)-1)
        start=max(0,endpoint-c.sequence_length);prefix=ids[start:endpoint]
        return torch.tensor(prefix,dtype=torch.long)[None],ids[endpoint],dict(sample_id=sample,prefix_sha256=hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),task_id=task,position=len(prefix)-1,
            source_position=endpoint-1,context_start=start,sequence_length=len(prefix),dataset=self.c.dataset,split=split)


class ProbeModel(nn.Module):
    def __init__(self,c,vocab):
        super().__init__();ec=EMCConfig(latent_dim=c.latent_dim,vocab_size=vocab,max_sequence_length=c.sequence_length,
            module_hidden_dim=c.hidden_dim,attention_heads=c.heads,num_modules=len(c.families),modules_per_cycle=1,
            ssm_backend='parallel_scan',delta_backend='parallel_delta',delta_internal_dim=c.latent_dim,delta_heads=c.heads,delta_ffn_dim=c.hidden_dim)
        self.embedding=nn.Embedding(vocab,c.latent_dim);self.position=nn.Embedding(c.sequence_length,c.latent_dim)
        self.context=create_emc_module(ec,'gpt');self.experts=nn.ModuleList([create_emc_module(ec,f) for f in c.families]);self.norm=nn.LayerNorm(c.latent_dim);self.head=nn.Linear(c.latent_dim,vocab,bias=False)
    def embed(self,x):
        h=self.embedding(x)+self.position(torch.arange(x.shape[1],device=x.device));return h+self.context(h)
    def logits(self,h):return self.head(self.norm(h[:,-1]))


def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def run(c,root,run_id,cancel_path=None):
    c.validate()
    if not run_id or any(not (v.isascii() and (v.isalnum() or v in '-_')) for v in run_id):raise ValueError('Invalid run ID')
    root=Path(root)/run_id;root.mkdir(parents=True,exist_ok=False)
    write(root/'config.json',asdict(c));started=time.perf_counter();rng=random.Random(c.seed+500);reservoir=[];seen=0;observations=0;measured=0;feature_seconds=0.;expert_seconds=0.;training_targets=0;training_context=0
    excluded_prefixes=set();evaluation_prefixes=set();duplicate_skips=0
    def check():
        if (root/'cancel.requested').exists() or (cancel_path and Path(cancel_path).exists()):raise InterruptedError('Cancelled at safe probe boundary')
    def emit(phase,**kw):
        event=dict(type='token_lab_progress',run_id=run_id,phase=phase,elapsed_seconds=time.perf_counter()-started,observations=observations,measured_locations=measured,**kw)
        print(json.dumps(event,allow_nan=False),flush=True);write(root/'status.json',dict(status='running',**event))
        with (root/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
    write(root/'feature-schema.json',list(mf.SCHEMA.values()))
    metadata=dict(schema_version=1,experiment_type='token_lab',run_id=run_id,started_at=datetime.now(timezone.utc).isoformat(),
        training_protocol='fresh common causal GPT context encoder + shared readout + independent expert parameters; equal mean endpoint losses for baseline and every expert/stage; no router',
        observation_unit='one endpoint per prefix; target withheld from all expert inputs',
        novelty='post-training measurements on training/reference inputs, not a record of training-time trajectories',
        sampling='Bernoulli locations; uniform reservoir of whole locations for bounded analysis; all expert rows kept together',
        independence='TinyStories validation blocks nonoverlapping but can share a document; capability examples distinct indices, may share templates',torch_version=torch.__version__)
    write(root/'metadata.json',metadata)
    try:
        torch.manual_seed(c.seed);torch.set_num_threads(1);emit('loading dataset');ds=Dataset(c);model=ProbeModel(c,ds.tokenizer.vocab_size).to(c.device)
        metadata.update(tokenizer=ds.tokenizer.to_config(),total_parameters=sum(p.numel() for p in model.parameters()),expert_parameters=[sum(p.numel() for p in e.parameters()) for e in model.experts],
            frequency_source='TinyStories train split' if c.dataset=='tinystories' else '1000 train-split capability examples, indices 0..999',
            frequency_sha256=hashlib.sha256(ds.counts.numpy().tobytes()).hexdigest())
        write(root/'metadata.json',metadata)
        calibration=None
        if c.experiment_mode=='forced_specialization':
            from .token_lab_calibration import CalibrationDataset,train_specialists
            ds=CalibrationDataset(c,ds)
            calibration,training_targets,training_context=train_specialists(c,model,ds,root,excluded_prefixes,check,emit)
            metadata['training_protocol']='controlled cloned specialists; frozen shared machinery; answer-endpoint supervision; see specialization.json'
            write(root/'metadata.json',metadata)
        opt=torch.optim.AdamW(model.parameters(),lr=c.learning_rate,weight_decay=0)
        for step in range(c.train_steps if calibration is None else 0):
            check();model.train();opt.zero_grad(set_to_none=True);total=0.
            for b in range(c.batch_size):
                check();x,target,meta=ds.sample('train',step*c.batch_size+b);excluded_prefixes.add(meta['prefix_sha256']);x=x.to(c.device);y=torch.tensor([target],device=c.device);h=model.embed(x)
                losses=[F.cross_entropy(model.logits(h),y)];state=h
                for expert in model.experts:
                    state=(h if c.topology=='independent' else state);out=state+expert(state);losses.append(F.cross_entropy(model.logits(out),y));state=out
                loss=torch.stack(losses).mean()/c.batch_size;loss.backward();total+=float(loss.detach());training_targets+=1;training_context+=x.numel()
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if step%10==0 or step+1==c.train_steps:emit('training',step=step+1,total=c.train_steps,loss=total,training_targets=training_targets,training_context_tokens=training_context)
        model.eval().requires_grad_(False);model.zero_grad(set_to_none=True)
        torch.save(dict(config=asdict(c),model=model.state_dict(),tokenizer=ds.tokenizer.to_config(),excluded_prefixes=sorted(excluded_prefixes)),root/'model.pt')
        if calibration is not None:
            from .token_lab_calibration import sanity_check
            calibration['sanity']=sanity_check(c,model,ds,excluded_prefixes,check,emit)
            write(root/'specialization.json',calibration)
        def observe(split,index,reference=False):
            nonlocal feature_seconds,expert_seconds,duplicate_skips
            check();x,target,meta=ds.sample(split,index);x=x.to(c.device)
            key=meta['prefix_sha256']
            if reference:excluded_prefixes.add(key)
            else:
                if key in excluded_prefixes or key in evaluation_prefixes:
                    duplicate_skips+=1;return None,None
                evaluation_prefixes.add(key)
            with torch.no_grad():common=model.embed(x)
            state=common;rows=[];hidden=[];losses=[];baselines=[]
            use_spectral=c.spectral_enabled and rng.random()<c.spectral_rate and not reference
            use_deep=c.deep_enabled and rng.random()<c.deep_rate and not reference
            for i,expert in enumerate(model.experts):
                check();before=common if c.topology=='independent' else state;t0=time.perf_counter()
                if index%10==0:emit('expert probe',expert_id=f'{i+1}:{c.families[i]}',expert_stage=i+1,sample_id=meta['sample_id'],split=split)
                feats=mf.core(before[0,-c.window:],c.knn_k)
                with torch.no_grad():pred=mf.prediction(model.logits(before)[0],target)
                delta,internal=mf.instrumentation(expert,before);expert_seconds+=internal.pop('_instrumented_expert_seconds');state=before+delta;after=mf.core(state[0,-c.window:],c.knn_k)
                with torch.no_grad():post=mf.prediction(model.logits(state)[0],target)
                if use_spectral:
                    spec=mf.spectral(before[0,-c.window:]);spec_after=mf.spectral(state[0,-c.window:]);feats.update(spec);after.update(spec_after)
                row={k:None for k in mf.SCHEMA};row.update(feats);row.update(pred);row.update(internal)
                row.update({'change_'+k:mf.scalar(v-(feats|pred)[k]) if v is not None and (feats|pred).get(k) is not None else None for k,v in (after|post).items() if 'change_'+k in mf.SCHEMA})
                h=before[0,-1].detach().cpu();d=delta[0,-1].detach().cpu();a=state[0,-1].detach().cpu()
                row.update(delta_norm=mf.scalar(d.norm()),relative_delta=mf.scalar(d.norm()/(h.norm()+mf.EPS)),input_output_distance=mf.scalar((a-h).norm()),input_delta_cosine=mf.cosine(h,d),input_output_cosine=mf.cosine(h,a),norm_change=mf.scalar(a.norm()-h.norm()))
                if use_deep:
                    z=before.detach().requires_grad_(True);loss=F.cross_entropy(model.logits(z),torch.tensor([target],device=c.device));g=torch.autograd.grad(loss,z)[0][0,-1].detach().cpu()
                    row.update(gradient_norm=mf.scalar(g.norm()),gradient_abs_mean=mf.scalar(g.abs().mean()),gradient_abs_max=mf.scalar(g.abs().max()),gradient_delta_cosine=mf.cosine(g,d),negative_gradient_delta_cosine=mf.cosine(-g,d))
                token=int(x[0,-1]);freq=(int(ds.counts[token])+1)/(int(ds.counts.sum())+len(ds.counts))
                row.update(run_id=run_id,**meta,expert_id=f'{i+1}:{c.families[i]}',expert_family=c.families[i],expert_stage=i+1,
                    topology=c.topology,window_size=min(c.window,x.shape[1]),token_id=token,token_text=ds.tokenizer.decode([token]),target_id=target,target_text=ds.tokenizer.decode([target]),
                    position_normalized=meta['position']/c.sequence_length,token_count=int(ds.counts[token]),token_frequency=freq,log_token_frequency=math.log(freq),rarity=-math.log(freq),token_char_length=len(ds.tokenizer.decode([token])),tie_epsilon=c.tie_epsilon)
                rows.append(row);hidden.append(h);losses.append(post['baseline_loss']);baselines.append(pred['baseline_loss']);feature_seconds+=time.perf_counter()-t0
            outs=mf.outcomes(baselines,losses,c.topology=='independent',c.tie_epsilon)
            for row,out in zip(rows,outs):
                row.update(out)
                row['relative_improvement']=row['improvement']/(abs(row['baseline_loss'])+mf.EPS)
                row['successful']=float(row['improvement']>0)
            return rows,hidden
        # Same deterministic reference inputs for every expert; no evaluation references.
        reference=[]
        for j in range(c.reference_size):
            rr,hh=observe('train',c.train_steps*c.batch_size+10000+j,True);reference.append((rr,hh))
            if j%10==0:emit('training reference bank',step=j+1,total=c.reference_size)
        banks=[]
        for i in range(len(c.families)):
            valid=[(rr[i],hh[i]) for rr,hh in reference if all(rr[i].get(k) is not None for k in mf.NOVEL_DESCRIPTOR)]
            values=torch.tensor([[r[k] for k in mf.NOVEL_DESCRIPTOR] for r,h in valid]);values=values.reshape(-1,len(mf.NOVEL_DESCRIPTOR))
            mean=values.mean(0) if len(values) else torch.zeros(len(mf.NOVEL_DESCRIPTOR));std=values.std(0,unbiased=False).clamp_min(.001) if len(values) else torch.ones_like(mean)
            z=(values-mean)/std;success=torch.tensor([r['improvement']>c.tie_epsilon for r,h in valid],dtype=torch.bool)
            hs=[F.normalize(h,dim=0) for rr,hh in reference for h in [hh[i]] if h.norm()>mf.EPS]
            banks.append(dict(hidden=torch.stack(hs) if hs else torch.empty(0,c.latent_dim),descriptor=z,mean=mean,std=std,success=z[success]))
        torch.save(dict(banks=banks,descriptor_keys=mf.NOVEL_DESCRIPTOR,reference_ids=[rr[0]['sample_id'] for rr,hh in reference],excluded_training_prefix_hashes=sorted(excluded_prefixes),token_counts=ds.counts),root/'reference-bank.pt')
        del reference
        sums=[dict(n=0,loss=0.,improvement=0.) for _ in c.families]
        raw=(root/'observations.jsonl').open('w',encoding='utf-8') if c.save_raw else None
        try:
            for j in range(c.evaluation_samples):
                check()
                if rng.random()>c.measurement_rate:continue
                rr,hh=observe('validation',j)
                if rr is None:continue
                for i,row in enumerate(rr):
                    row.update(mf.novelty(hh[i],row,banks[i],c.knn_k));row['unavailable_features']=[k for k in mf.SCHEMA if row.get(k) is None]
                    if raw:raw.write(json.dumps(row,allow_nan=False)+'\n')
                    sums[i]['n']+=1;sums[i]['loss']+=row['expert_loss'];sums[i]['improvement']+=row['improvement']
                observations+=len(rr);measured+=1;seen+=1
                if len(reservoir)<c.analysis_cap:reservoir.append(rr)
                else:
                    pick=rng.randrange(seen)
                    if pick<c.analysis_cap:reservoir[pick]=rr
                if j%10==0:emit('evaluation',step=j+1,total=c.evaluation_samples)
        finally:
            if raw:raw.close()
        emit('analysis');rows=[r for group in reservoir for r in group]
        analysis=analyze(rows,c.seed,lambda:(root/'cancel.requested').exists() or bool(cancel_path and Path(cancel_path).exists()));check()
        if calibration is not None:
            from .token_lab_calibration_analysis import calibration_analysis
            calibration['sampling_rejections']=dict(short_examples=ds.short_rejections,overlapping_prefixes=ds.overlap_rejections)
            write(root/'specialization.json',calibration)
            analysis['calibration']=calibration_analysis(rows,c,calibration,check)
            write(root/'calibration-analysis.json',analysis['calibration'])
        summary=dict(run_id=run_id,name=c.name,status='completed',experiment_type='token_lab',schema_version=1,started_at=metadata['started_at'],
            observations=observations,measured_locations=measured,duplicate_prefixes_skipped=duplicate_skips,analysis_locations=len(reservoir),analysis_sampling='uniform whole-location reservoir after exact-prefix exclusions; inclusion=min(1,analysis_cap/measured_locations)',
            raw_saved=c.save_raw,display_rows=min(10000,len(rows)),overall_full_stream=[dict(expert_id=f'{i+1}:{c.families[i]}',n=s['n'],mean_loss=s['loss']/s['n'] if s['n'] else None,mean_improvement=s['improvement']/s['n'] if s['n'] else None) for i,s in enumerate(sums)],
            training_targets=training_targets,training_context_tokens=training_context,runtime_seconds=time.perf_counter()-started,probe_and_measurement_seconds=feature_seconds,
            instrumented_expert_seconds=expert_seconds,feature_and_diagnostic_seconds=max(0.,feature_seconds-expert_seconds),
            measurement_overhead='feature_and_diagnostic_seconds excludes instrumented expert forward; FFN hook overhead remains in expert seconds. Includes reference and evaluation measurement, not an uninstrumented wall-clock A/B benchmark.',locations_per_second=measured/max(time.perf_counter()-started,1e-8))
        write(root/'analysis.json',analysis);write(root/'display.json',rows[:10000]);write(root/'summary.json',summary)
        report_text=report(asdict(c),summary,analysis)
        if calibration is not None:
            from .token_lab_calibration_analysis import calibration_report
            from .token_lab_calibration_analysis import sweep_record
            summary['calibration']=sweep_record(c,calibration,analysis['calibration'])
            sweep=[]
            for path in root.parent.glob('*/summary.json'):
                check()
                if path.parent==root:continue
                try:other=json.loads(path.read_text(encoding='utf-8'))
                except (ValueError,OSError):continue
                if other.get('status')=='completed' and other.get('calibration',{}).get('comparison_key')==summary['calibration']['comparison_key']:
                    sweep.append(dict(run_id=other['run_id'],**other['calibration']))
            sweep.append(dict(run_id=run_id,**summary['calibration']));sweep.sort(key=lambda r:r['strength'])
            analysis['calibration']['strength_sweep']=sweep
            write(root/'analysis.json',analysis);write(root/'calibration-analysis.json',analysis['calibration'])
            write(root/'summary.json',summary)
            report_text=calibration_report(c,calibration,analysis['calibration'])+'\n\n# Secondary standard Token Lab analysis (not blind)\n\n'+report_text
        (root/'report.md').write_text(report_text,encoding='utf-8');write(root/'status.json',dict(status='completed'));emit('complete')
        write(root/'status.json',dict(status='completed'))
        return summary
    except BaseException as exc:
        status='cancelled' if isinstance(exc,InterruptedError) else 'failed';write(root/'status.json',dict(status=status,error=str(exc)));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['schema','estimate','validate','run']);p.add_argument('config',nargs='?');p.add_argument('--runs-dir',default='token-lab-runs');p.add_argument('--run-id',default=str(int(time.time())));args=p.parse_args()
    if args.command=='schema':print(json.dumps(dict(defaults=asdict(LabConfig()),families=['gpt','ssm','recurrent','delta'],tasks=list(CAPABILITIES))));return
    request=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if request.get('experiment_mode')=='frozen_routing':
        from .token_lab_routing import validate,run_test,sources
        request=validate(request)
        if args.command in ['estimate','validate']:
            print(json.dumps(dict(valid=True,warning=f"{len(sources(request))} saved expert banks; one-step and {request['sequential_steps']}-step routing; simple/full features; no expert retraining.")));return
        run_test(request,args.runs_dir,args.run_id);return
    if request.get('experiment_mode')=='validation_sweep':
        from .token_lab_sweep import validate,run_sweep
        request=validate(request)
        if args.command in ['estimate','validate']:
            print(json.dumps(dict(valid=True,warning=f"Automated sweep: {8*request['sweep_seed_count']} calibration runs plus analyses; common bases and controls are managed automatically.")));return
        run_sweep(request,args.runs_dir,args.run_id);return
    if request.get('analysis_only'):
        from .token_lab_discovery import validate,run_saved
        validate(request)
        if args.command in ['estimate','validate']:
            print(json.dumps(dict(valid=True,warning='Saved-observation analysis only; no training or expert execution.')));return
        run_saved(request,args.runs_dir,args.run_id);return
    c=LabConfig(**request).validate()
    if args.command in ['estimate','validate']:print(json.dumps(dict(valid=True,config=asdict(c),warning='Fresh standalone training, not your EMC checkpoint. Every expert receives equal training opportunities.')));return
    run(c,args.runs_dir,args.run_id)


if __name__=='__main__':main()
