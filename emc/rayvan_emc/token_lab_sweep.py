"""One managed, restartable research sweep. Reuses Token Lab; never trains a router."""
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import io
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from .token_lab import LabConfig, run, write
from . import token_lab_discovery as discovery
from .token_lab_calibration_analysis import split_rows, regression_scores

FAMILIES = ('gpt', 'ssm', 'recurrent', 'delta')
DEFAULTS = dict(experiment_mode='validation_sweep', name='Specialization Validation Sweep',
    sweep_seed_start=42, sweep_seed_count=3, diagnostic_seed=42,
    common_pretrain_steps=1000, train_steps=1000, batch_size=4, learning_rate=.001,
    evaluation_samples=2000, calibration_samples_per_task=50, latent_dim=64,
    hidden_dim=128, heads=4, sequence_length=48, window=16, reference_size=128,
    spectral_enabled=True, spectral_rate=1., deep_enabled=False, deep_rate=.05,
    device='cpu', max_features=5, candidate_count=16, cross_task=True, resume_from='')


def child_config(c, family, seed, strength, checkpoint=''):
    fields = {k: c[k] for k in DEFAULTS if k in LabConfig.__dataclass_fields__}
    return LabConfig(**(fields | dict(name=f'{family.upper()} · {strength*100}% · seed {seed}',
        experiment_mode='forced_specialization', dataset='capability_10', families=(family, family),
        topology='independent', seed=seed, specialization_strength=float(strength),
        common_base_source='checkpoint' if checkpoint else 'pretrain', common_checkpoint=checkpoint,
        specialist_checkpoint='', specialization_profile='two_way', task_weights=None,
        identical_stream=True, save_raw=True, measurement_rate=1., analysis_cap=c['evaluation_samples'])))


def signature(c):
    return {k:v for k,v in c.items() if k not in ['name','resume_from']}


def validate(request):
    # Deliberately ignore stale manual Token Lab checkpoint/family/strength fields.
    c = {k:request.get(k,v) for k,v in DEFAULTS.items()}
    for key,lo,hi in [('sweep_seed_count',1,20),('sweep_seed_start',0,2**31-21),
                       ('diagnostic_seed',0,2**31-1),('max_features',1,10),('candidate_count',2,40)]:
        if type(c[key]) is not int or not lo<=c[key]<=hi:raise ValueError(f'{key} must be an integer in [{lo},{hi}]')
    if c['candidate_count']<c['max_features']:raise ValueError('Candidate count must cover selected features')
    if c['evaluation_samples']<100:raise ValueError('Use at least 100 evaluation locations per run')
    for key in ['cross_task','spectral_enabled','deep_enabled']:
        if type(c[key]) is not bool:raise ValueError(f'{key} must be a boolean')
    for family in FAMILIES:child_config(c,family,c['sweep_seed_start'],1).validate()
    if c['resume_from']:
        source=Path(c['resume_from'])
        old=json.loads((source/'config.json').read_text(encoding='utf-8'))
        if old.get('experiment_mode')!='validation_sweep' or signature(old)!=signature(c):
            raise ValueError('Resume requires the same sweep settings. Load the saved sweep settings first.')
        if not (source/'sweep-manifest.json').is_file():raise ValueError('Resume sweep manifest missing')
    return c


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checkpoint_json(path,obj):
    """Keep parent resume metadata readable if the process exits during a write."""
    temporary=path.with_suffix(path.suffix+'.tmp')
    write(temporary,obj);temporary.replace(path)


def verify_controls(path, counterpart=None):
    info=load(path/'specialization.json')
    if not info['clone_identity_verified'] or len(set(info['initial_hashes']))!=1 or not info['shared_unchanged']:
        raise ValueError('Calibration clone/freeze verification failed')
    budgets=[{k:b[k] for k in ['steps','samples','target_tokens','context_tokens']} for b in info['budgets']]
    if budgets[0]!=budgets[1]:raise ValueError('Specialist training budgets differ')
    if counterpart:
        other=load(counterpart/'specialization.json')
        for key in ['common_hash','shared_hash','initial_hashes','expert_parameter_counts','optimizer']:
            if info[key]!=other[key]:raise ValueError('Matched control differs: '+key)
        if budgets!=[{k:b[k] for k in budgets[0]} for b in other['budgets']]:raise ValueError('Strength budgets differ')
    return dict(common_hash=info['common_hash'],initial_hashes=info['initial_hashes'],shared_unchanged=True,
        equal_budgets=True,exact_control_final_identity=info['exact_control_final_identity'],
        strength=info['strength'],budgets=budgets,expert_parameter_counts=info['expert_parameter_counts'])


def best_fit(X,y,keys,masks,indices,seed,check):
    candidates=[]
    for kind in ['ridge','mlp']:
        p,meta=discovery.fit(X,y,masks,indices,kind,seed,check)
        candidates.append((float(np.mean((p[masks[1]]-y[masks[1]])**2)),p,meta))
    _,p,meta=min(candidates,key=lambda x:x[0])
    return p,meta


def group_indices(keys):
    controls={'source_position','context_start','token_count','token_frequency','log_token_frequency',
              'rarity','predictive_entropy','top1_probability','logit_margin','position','position_normalized','token_char_length'}
    spectral=lambda k:k.startswith(('spectral_','laplacian_'))
    novelty=lambda k:'novelty' in k or k=='reference_max_cosine'
    return {'controls':[i for i,k in enumerate(keys) if k in controls],
            'all':list(range(len(keys))),
            'without_geometry':[i for i,k in enumerate(keys) if k in controls or novelty(k)],
            'without_spectral':[i for i,k in enumerate(keys) if not spectral(k)],
            'without_novelty':[i for i,k in enumerate(keys) if not novelty(k)]}


def decision_scores(rows,y,p,masks,seed):
    tr,va,te=masks
    # Positive y = expert 1 advantage. Constant choice is determined on TRAIN only.
    regret=np.maximum(y[te],0)-y[te]*(p[te]>0)
    constant=np.maximum(y[te],0)-y[te]*(y[tr].mean()>0)
    delta=constant-regret
    valid=abs(y[te])>.001
    testrows=[r for r,t in zip(rows,te) if t]
    return dict(mean_regret=float(regret.mean()),constant_regret=float(constant.mean()),
        uniform_expected_regret=float(np.mean(abs(y[te]))/2),gain_over_constant=float(delta.mean()),
        gain_ci=discovery.paired_interval(testrows,delta,seed),
        non_tie_accuracy=float(np.mean((p[te][valid]>0)==(y[te][valid]>0))) if valid.any() else None,
        n=int(te.sum()),non_tie_n=int(valid.sum()),meaning='Offline single-step decision diagnostic, not a sequential EMC router test')


def extra_analysis(path,c,check):
    locations,_,_=discovery.read_locations(path/'observations.jsonl',c['evaluation_samples'],c['diagnostic_seed'],check,True)
    datasets=discovery.targets(locations)
    pair=next(rows for name,rows in datasets.items() if name.endswith(' advantage'))
    data=discovery.prepare(pair)
    if data is None:return dict(status='insufficient split sizes')
    X,y,keys,masks=data
    if np.max(abs(y))<1e-10:return dict(status='identical expert outcomes; differential probes skipped')
    if np.var(y[masks[0]])<1e-12:return dict(status='constant training advantage; differential probes skipped')
    groups={};preds={};seed=c['diagnostic_seed'];tr,va,te=masks
    for name,indices in group_indices(keys).items():
        check();p,meta=best_fit(X,y,keys,masks,indices,seed,check);preds[name]=p
        groups[name]=dict(features=[keys[i] for i in indices],model=meta['kind'],
            scores=regression_scores(y[te],p[te]),decisions=decision_scores(pair,y,p,masks,seed))
    testrows=[r for r,t in zip(pair,te) if t]
    for name in groups:
        if name=='all':continue
        delta=abs(preds[name][te]-y[te])-abs(preds['all'][te]-y[te])
        groups[name]['mae_minus_all']=float(delta.mean())
        groups[name]['mae_minus_all_ci']=discovery.paired_interval(testrows,delta,seed)
    # Nonconstant shuffled-label negative check, in addition to the identical-expert control.
    rng=np.random.default_rng(seed);shuffled=y.copy()
    for mask in masks:shuffled[mask]=rng.permutation(y[mask])
    p,meta=best_fit(X,shuffled,keys,masks,list(range(len(keys))),seed,check)
    null=dict(scores=regression_scores(shuffled[te],p[te]),model=meta['kind'],
        note='One split-wise label permutation, a sanity check not a formal permutation p-value')
    cross=[]
    if c['cross_task']:
        task_names=sorted({r.get('task_id') for r in pair if r.get('task_id')})
        hashed=split_rows(pair)
        for task in task_names:
            check()
            # Task identity only defines the exclusion split; never enters X.
            split=np.array([8 if r.get('task_id')==task else (0 if h<8 else 6) for r,h in zip(pair,hashed)])
            prepared=discovery.prepare(pair,split)
            if prepared is None:cross.append(dict(task=task,status='insufficient split sizes'));continue
            xx,yy,kk,mm=prepared
            entry=dict(task=task,status='ok',train_n=int(mm[0].sum()),validation_n=int(mm[1].sum()),test_n=int(mm[2].sum()))
            for name,ii in group_indices(kk).items():
                if name not in ['all','controls']:continue
                pp,meta=best_fit(xx,yy,kk,mm,ii,seed,check)
                entry[name]=dict(scores=regression_scores(yy[mm[2]],pp[mm[2]]),model=meta['kind'],
                    decisions=decision_scores(pair,yy,pp,mm,seed))
            cross.append(entry)
    return dict(status='ok',groups=groups,shuffled_label_check=null,cross_task=cross,
        note='Model family chosen using validation only. Pointwise bootstrap intervals are exploratory and conditional on fitted models. No multiplicity correction. Cross-task means within this synthetic suite, not real-world transfer.')


def aggregate(entries):
    families={}
    for family in FAMILIES:
        rows=[e for e in entries if e['family']==family and e.get('status')=='completed']
        positives=[e for e in rows if e['strength']==1];controls={e['seed']:e for e in rows if e['strength']==0}
        paired=[e for e in positives if e['seed'] in controls and controls[e['seed']]['controls']['exact_control_final_identity']]
        good=[e for e in paired if e.get('extra',{}).get('status')=='ok']
        gains=[e['extra']['groups']['all']['decisions']['gain_over_constant'] for e in good]
        formed=[all(v['specialization_formed'] for v in e['sanity'].values()) for e in good]
        group_evidence={}
        for group in ['controls','without_geometry','without_spectral','without_novelty']:
            evidence=[e['extra']['groups'][group] for e in good
                if e['extra']['groups'][group]['features']!=e['extra']['groups']['all']['features']]
            group_evidence[group]=dict(tested_seeds=len(evidence),
                positive_mae_gains=sum(g['mae_minus_all']>0 for g in evidence),
                positive_intervals=sum(g['mae_minus_all_ci'][0]>0 for g in evidence),
                negative_intervals=sum(g['mae_minus_all_ci'][1]<0 for g in evidence),
                mean_mae_gain=float(np.mean([g['mae_minus_all'] for g in evidence])) if evidence else None)
        if not good:verdict='INCONCLUSIVE: no complete eligible pairs'
        elif not all(formed):verdict='INCONCLUSIVE: specialization sanity check failed in some replicates'
        elif len(paired)!=len(positives):verdict='INCONCLUSIVE: missing or non-identical controls'
        elif len(good)<3:verdict='PRELIMINARY: fewer than three matched seeds'
        elif any((e['extra']['shuffled_label_check']['scores']['r2'] or 0)>.1 for e in good):verdict='INCONCLUSIVE: unexpectedly predictive shuffled-label check'
        elif all(g>0 for g in gains):verdict='SUPPORTS offline state-dependent choice across tested seeds'
        else:verdict='INCONSISTENT offline choice benefit across tested seeds'
        features={}
        for e in positives:
            for target,d in e['discovery'].items():
                for a in d.get('ablations',[]):
                    key=target+' / '+a['feature'];f=features.setdefault(key,dict(selected=0,positive_removal_intervals=0,negative_removal_intervals=0,associations=[]))
                    f['selected']+=1;f['positive_removal_intervals']+=int(a['mae_increase_ci'][0]>0);f['negative_removal_intervals']+=int(a['mae_increase_ci'][1]<0)
                    assoc=next((v for v in d.get('held_out_associations',[]) if v['feature']==a['feature']),{})
                    if assoc.get('spearman') is not None:f['associations'].append(assoc['spearman'])
        families[family]=dict(completed_runs=len(rows),matched_identical_control_pairs=len(paired),eligible_replicates=len(good),
            verdict=verdict,choice_gain_per_seed=gains,
            mean_choice_gain=float(np.mean(gains)) if gains else None,
            min_choice_gain=min(gains) if gains else None,max_choice_gain=max(gains) if gains else None,
            feature_recurrence=features,feature_denominator=len(positives),group_evidence=group_evidence)
    return families


def make_report(result):
    lines=['# Specialization Validation Sweep',
        'Controlled calibration, not emergent specialization or a production router. Four module families; two homogeneous experts per run.',
        'Each family/seed has a newly pretrained base reused exactly at 100% and 0%. Expert-only updates; equal budgets; no task labels in predictors.',
        'Families have different parameter counts and their own bases; comparisons are not architecture-isolated causal effects.',
        'Generated-example-group train/validation/test splits; target-free pre-expert features only. Novelty references use training data.',
        '0%/100% evaluation sets can differ due to training-prefix exclusions; across-strength scores are NOT paired observations.',
        'Three or more seeds are a screening minimum, not a guarantee. No universal signature or once-and-for-all validation is claimed.',
        f"Status: {result['status']}. Completed {sum(e.get('status')=='completed' for e in result['entries'])}/{result['planned_runs']}.",
        '\n## Replication overview','| Family | Matched controls | Mean offline choice gain | Verdict |','|---|---:|---:|---|']
    for family,a in result['families'].items():lines.append(f"| {family} | {a['matched_identical_control_pairs']} | {a['mean_choice_gain']} | {a['verdict']} |")
    lines+=['\nChoice gain is constant-expert loss minus predicted-choice loss on held-out states, in nats; positive is better. The constant expert is chosen on training data. This does not test sequential trajectories.',
        '\n## Individual experiments','| Family | Seed | Strength | Outcome | All MLP R² | Compact MLP R² |','|---|---:|---:|---|---:|---:|']
    provenance=[]
    for e in result['entries']:
        if e.get('status')!='completed':lines.append(f"| {e['family']} | {e['seed']} | {e['strength']} | {e.get('error',e['status'])} | — | — |");continue
        for target,d in e['discovery'].items():
            scores=d.get('scores',{})
            lines.append(f"| {e['family']} | {e['seed']} | {e['strength']} | {target}: {d['status']} | {scores.get('all_mlp',{}).get('r2')} | {scores.get('compact_mlp',{}).get('r2')} |")
        provenance.append(f"- {e['family']} seed {e['seed']} strength {e['strength']}: common hash {e['controls']['common_hash']}; exact final clone identity {e['controls']['exact_control_final_identity']}; raw SHA {e['raw_sha256']}")
    lines+=['\n## Provenance',*provenance]
    lines+=['\n## Feature recurrence (100% runs)', 'Selection frequency is not independent importance; correlated substitutes and limited screening affect it. Intervals are pointwise, not multiplicity-adjusted.']
    for family,a in result['families'].items():
        lines.append(f"\n### {family} — {a['feature_denominator']} completed specialization replicates")
        for group,g in a['group_evidence'].items():
            lines.append(f"- All inputs versus {group}: {g['tested_seeds']} tested seeds; full-set MAE benefit in {g['positive_mae_gains']}; positive intervals {g['positive_intervals']}; negative intervals {g['negative_intervals']}; mean benefit {g['mean_mae_gain']}")
        for feature,f in sorted(a['feature_recurrence'].items(),key=lambda kv:(-kv[1]['positive_removal_intervals'],-kv[1]['selected'],kv[0])):
            lines.append(f"- {feature}: selected {f['selected']}; positive removal intervals {f['positive_removal_intervals']}; negative {f['negative_removal_intervals']}; test Spearman {f['associations']}")
    lines+=['\n## Pre-specified feature-group and cross-task checks',
        'Controls = position, frequency and prediction confidence. Geometry block = remaining hidden/dynamics/local-shape/spectral features. Novelty is separate. Positive MAE-minus-all means the full set improved prediction over that reduced set.']
    for e in result['entries']:
        if e.get('status')!='completed' or e.get('extra',{}).get('status')!='ok':continue
        lines.append(f"\n### {e['family']} seed {e['seed']} strength {e['strength']}")
        for name,g in e['extra']['groups'].items():lines.append(f"- {name}: test R² {g['scores']['r2']}; MAE {g['scores']['mae']}; MAE-minus-all {g.get('mae_minus_all')}, CI {g.get('mae_minus_all_ci')}; choice gain {g['decisions']['gain_over_constant']}, CI {g['decisions']['gain_ci']}")
        lines.append(f"- Shuffled-label sanity check: {e['extra']['shuffled_label_check']}")
        for t in e['extra']['cross_task']:
            if t['status']!='ok':lines.append(f"- Held-out {t['task']}: {t['status']}");continue
            lines.append(f"- Excluded task {t['task']}, n={t['test_n']}: all-input R² {t['all']['scores']['r2']}; controls R² {t['controls']['scores']['r2']}; all-input choice gain {t['all']['decisions']['gain_over_constant']}")
        for expert,s in e['sanity'].items():lines.append(f"- Sanity {expert}: formed={s['specialization_formed']}; differential gap={s['differential_gap']}; CI={s['differential_gap_ci']}")
    lines+=['\n## Negative evidence and limits',
        'Inspect nonpositive group gains, removal intervals crossing zero, negative held-out R², failed task transfers and controls. They are retained, never filtered out.',
        'A constant control is deliberately skipped, not proof against arbitrary false discoveries; the shuffled-label check is only one permutation.',
        'A failed specialization sanity check makes descriptor failure uninterpretable. A positive improvement probe alone does not demonstrate specialization.',
        'Feature sets are selected independently per replicate. This tests the discovery procedure, not transfer of one frozen predictor or feature set to a new seed.',
        'Use cross-task and group checks to decide whether geometry adds beyond task-format/confidence clues. Do not infer a sequential EMC benefit from offline choice gains.',
        'All completed, failed and cancelled jobs remain in sweep-manifest.json. Child checkpoints, observations and reports live under jobs/.']
    return '\n'.join(lines)


class Relay(io.TextIOBase):
    def __init__(self,emit):self.emit=emit;self.buffer=''
    def write(self,s):
        self.buffer+=s
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1)
            if not line.strip():continue
            try:event=json.loads(line)
            except ValueError:continue
            self.emit(event)
        return len(s)
    def flush(self):pass


def run_sweep(request,runs_root,run_id):
    c=validate(request)
    if not run_id or any(not (v.isascii() and (v.isalnum() or v in '-_')) for v in run_id):raise ValueError('Invalid run ID')
    root=(Path(runs_root)/run_id).resolve();root.mkdir(parents=True,exist_ok=False);jobs=root/'jobs';jobs.mkdir()
    stdout=sys.stdout;start=time.perf_counter();started=datetime.now(timezone.utc).isoformat();entries=[]
    result=dict(schema_version=1,status='running',settings=c,planned_runs=8*c['sweep_seed_count'],entries=entries,families={})
    write(root/'config.json',c)
    write(root/'metadata.json',dict(started_at=started,experiment_type='token_lab_sweep',protocol_version=1,torch_version=torch.__version__))
    previous={}
    if c['resume_from']:
        previous={e['key']:e for e in load(Path(c['resume_from'])/'sweep-manifest.json')['entries']}
    def check():
        if (root/'cancel.requested').exists():raise InterruptedError('Sweep cancelled; completed jobs are preserved')
    def emit(phase,**kw):
        event=dict(type='token_lab_progress',run_id=run_id,phase=phase,elapsed_seconds=time.perf_counter()-start,
            completed_jobs=sum(e.get('status')=='completed' for e in entries),planned_jobs=result['planned_runs'],**kw)
        write(root/'status.json',dict(status='running',**event));print(json.dumps(event),file=stdout,flush=True)
        with (root/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
    def persist(status):
        result['status']=status;result['families']=aggregate(entries)
        checkpoint_json(root/'sweep-manifest.json',dict(schema_version=1,entries=[{k:v for k,v in e.items() if k not in ['discovery','sanity','extra']} for e in entries]))
        checkpoint_json(root/'sweep-analysis.json',result);checkpoint_json(root/'analysis.json',dict(sweep=result))
        checkpoint_json(root/'summary.json',dict(run_id=run_id,name=c['name'],status=status,started_at=started,
            experiment_type='token_lab_sweep',completed_jobs=sum(e.get('status')=='completed' for e in entries),
            planned_jobs=result['planned_runs'],runtime_seconds=time.perf_counter()-start))
        (root/'report.md').write_text(make_report(result),encoding='utf-8')
    persist('running')
    try:
        for family in FAMILIES:
            for seed in range(c['sweep_seed_start'],c['sweep_seed_start']+c['sweep_seed_count']):
                base=None
                for strength in [1,0]:
                    check();key=f'{family}-seed-{seed}-s{strength*100}';old=previous.get(key,{})
                    entry=dict(key=key,family=family,seed=seed,strength=strength,status='running');entries.append(entry);persist('running')
                    try:
                        source=Path(old.get('source_directory',jobs/key))
                        cfg=child_config(c,family,seed,strength,str(base/'common-base.pt') if base else '')
                        if strength==0 and base is None:raise ValueError('100% job failed; common base unavailable for matched control')
                        expected=asdict(cfg)
                        complete=(source/'status.json').is_file() and load(source/'status.json').get('status')=='completed'
                        if complete:
                            actual=load(source/'config.json')
                            # A resumed base can live under an earlier sweep root.
                            if json.dumps(actual,sort_keys=True)!=json.dumps(expected,sort_keys=True):raise ValueError('Saved child configuration mismatch; refusing reuse')
                        else:
                            source=jobs/key
                            entry['source_directory']=str(source.resolve());persist('running')
                            def child_event(ev):emit(f"{key} · {ev.get('phase','working')}",**{k:v for k,v in ev.items() if k in ['step','total','expert_id','measured_locations','observations','loss']})
                            with redirect_stdout(Relay(child_event)):run(cfg,jobs,key,cancel_path=root/'cancel.requested')
                        entry['source_directory']=str(source.resolve())
                        entry['controls']=verify_controls(source,base if strength==0 else None)
                        if entry['controls']['strength']!=strength:raise ValueError('Recorded specialization strength differs from plan')
                        if strength==1:base=source
                        persist('running')
                        check();emit(f'{key} · saved feature discovery')
                        analysis_id=key+'-discovery';analysis_path=jobs/analysis_id
                        req=dict(analysis_only=True,source_run_directory=str(source.resolve()),max_features=c['max_features'],candidate_count=c['candidate_count'],location_cap=c['evaluation_samples'],seed=c['diagnostic_seed'])
                        with redirect_stdout(Relay(lambda ev:emit(f"{key} · {ev.get('phase','analysis')}"))):discovery.run_saved(req,jobs,analysis_id,cancel_path=root/'cancel.requested')
                        found=load(analysis_path/'discovery-analysis.json')
                        compact={}
                        for name,d in found['targets'].items():
                            compact[name]={k:v for k,v in d.items() if k in ['status','counts','constant','selected_features','scores','validation_chosen_model','ablations']}
                            compact[name]['held_out_associations']=[{k:v for k,v in a.items() if k in ['feature','n','pearson','spearman','interval']} for a in d.get('held_out_associations',[])]
                        sanity={name:{k:v for k,v in s.items() if k in ['task_matrix','specialization_gap','differential_gap','differential_gap_ci','specialization_formed']} for name,s in load(source/'calibration-analysis.json')['reveal'].items()}
                        entry.update(discovery=compact,raw_sha256=found['raw_sha256'],analysis_directory=str(analysis_path.resolve()),sanity=sanity)
                        emit(f'{key} · feature-group, shuffled-label and cross-task checks')
                        entry['extra']=extra_analysis(source,c,check)
                        entry['status']='completed'
                    except InterruptedError:
                        entry['status']='cancelled';raise
                    except Exception as exc:
                        entry.update(status='failed',error=str(exc))
                        with (root/'logs.txt').open('a',encoding='utf-8') as f:f.write(key+'\n'+traceback.format_exc()+'\n')
                        emit(f'{key} · failed; continuing remaining jobs')
                    finally:
                        gc.collect()
                        if torch.cuda.is_available():torch.cuda.empty_cache()
                        persist('running')
        status='completed' if all(e['status']=='completed' for e in entries) else 'partial'
        persist(status);emit('sweep finished');write(root/'status.json',dict(status=status));return result
    except BaseException as exc:
        status='cancelled' if isinstance(exc,InterruptedError) else 'failed';persist(status)
        write(root/'status.json',dict(status=status,error=str(exc)));raise
