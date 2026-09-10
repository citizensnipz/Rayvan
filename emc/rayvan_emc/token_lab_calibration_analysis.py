"""Blind diagnostic probes followed by label reveal. Never updates experts."""
import copy
import hashlib
import itertools
import json
import numpy as np
import torch
from torch import nn
from .token_lab_features import blind_keys, SCHEMA
from .token_lab_analysis import association, finite, corr, ranks


def split_rows(rows):
    # Group related endpoints of a generated example; same assignment for every expert.
    return np.array([int(hashlib.sha256(r.get('analysis_group',r['sample_id']).encode()).hexdigest()[:8],16)%10 for r in rows])


def regression_scores(y,p):
    den=np.square(y-y.mean()).sum()
    return dict(n=len(y),r2=float(1-np.square(y-p).sum()/den) if den>1e-12 else None,
                mae=float(abs(y-p).mean()),spearman=corr(ranks(y),ranks(p)))


def classification_scores(y,p):
    y=np.asarray(y,dtype=bool);n1=int(y.sum());n0=len(y)-n1
    if not n1 or not n0:return dict(n=len(y),status='one class; metrics undefined',roc_auc=None,pr_auc=None,balanced_accuracy=None)
    auc=(ranks(p)[y].sum()-n1*(n1-1)/2)/(n1*n0)
    # Average precision with ties grouped at identical thresholds, not arbitrary tie order.
    order=np.argsort(-p,kind='stable');labels=y[order];scores=p[order]
    ends=np.r_[np.flatnonzero(np.diff(scores)),len(scores)-1];tp=np.cumsum(labels)[ends];precision=tp/(ends+1);recall=tp/n1
    ap=np.sum(precision*np.diff(np.r_[0.,recall]));pred=p>=.5
    return dict(n=len(y),roc_auc=float(auc),pr_auc=float(ap),pr_auc_definition='noninterpolated average precision; tied scores grouped',
                prevalence=n1/len(y),balanced_accuracy=float((pred[y].mean()+(~pred[~y]).mean())/2))


def probe(rows,keys,seed,q,check):
    # Do not trust caller-provided feature lists.
    keys=[k for k in keys if k in blind_keys()]
    split=split_rows(rows);tr=split<6;va=(split>=6)&(split<8);te=split>=8
    if min(tr.sum(),va.sum(),te.sum())<15:return dict(status='insufficient train/validation/test groups',counts=[int(m.sum()) for m in [tr,va,te]])
    keys=[k for k in keys if np.mean([finite(r.get(k)) for r,t in zip(rows,tr) if t])>=.8]
    if not keys:return dict(status='no available input features')
    X=np.array([[r.get(k) if finite(r.get(k)) else np.nan for k in keys] for r in rows]);y=np.array([r['improvement'] for r in rows])
    med=np.nanmedian(X[tr],axis=0);X=np.where(np.isnan(X),med,X);mu=X[tr].mean(0);sd=X[tr].std(0);keep=sd>1e-8
    keys=[k for k,v in zip(keys,keep) if v];X=(X[:,keep]-mu[keep])/sd[keep]
    if not keys:return dict(status='constant features')
    center=y[tr].mean();scale=max(y[tr].std(),1e-5)
    options=[]
    for alpha in [1.,10.,100.,1000.]:
        check();coef=np.linalg.solve(X[tr].T@X[tr]+alpha*np.eye(len(keys)),X[tr].T@(y[tr]-center));pred=X@coef+center
        options.append((np.mean((pred[va]-y[va])**2),alpha,coef,pred))
    _,alpha,coef,pred=min(options,key=lambda v:v[0])
    result=dict(status='ok',features=keys,split='example-group SHA256 60/20/20; train preprocessing and success threshold; validation hyperparameters; untouched test metrics',
        ridge=dict(alpha=alpha,train=regression_scores(y[tr],pred[tr]),test=regression_scores(y[te],pred[te]),coefficients=dict(zip(keys,coef.tolist()))),
        mean_baseline=regression_scores(y[te],np.full(te.sum(),center)))
    # baseline_loss and baseline surprisal are the SAME quantity, not independent baselines.
    b=np.array([r['baseline_loss'] for r in rows]);bm=b[tr].mean();den=np.square(b[tr]-bm).sum()
    slope=float((b[tr]-bm)@(y[tr]-center)/den) if den>1e-12 else 0
    result['baseline_loss_only']=regression_scores(y[te],center+slope*(b[te]-bm));result['baseline_surprisal_only']='identical to baseline_loss_only'
    xt=torch.tensor(X,dtype=torch.float32);yt=torch.tensor((y-center)/scale,dtype=torch.float32)
    def fit(target,classification=False):
        torch.manual_seed(seed);net=nn.Sequential(nn.Linear(len(keys),32),nn.GELU(),nn.Linear(32,1))
        optimizer=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.01)
        best=float('inf');state=None;best_epoch=0
        target=torch.tensor(target,dtype=torch.float32)
        for epoch in range(150):
            check();net.train();optimizer.zero_grad();out=net(xt[tr]).squeeze(-1)
            loss=nn.functional.binary_cross_entropy_with_logits(out,target[tr]) if classification else nn.functional.mse_loss(out,target[tr])
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step();net.eval()
            with torch.no_grad():
                out=net(xt[va]).squeeze(-1);v=float(nn.functional.binary_cross_entropy_with_logits(out,target[va]) if classification else nn.functional.mse_loss(out,target[va]))
            if v<best:best=v;state=copy.deepcopy(net.state_dict());best_epoch=epoch
            if epoch-best_epoch>=20:break
        net.load_state_dict(state);net.eval()
        with torch.no_grad():p=net(xt).squeeze(-1);p=p.sigmoid() if classification else p*scale+center
        return p.numpy(),net,best_epoch+1
    mp,net,epoch=fit(yt.numpy())
    result['mlp']=dict(architecture='standardized inputs → 32 GELU → 1; AdamW .003, decay .01; max150 epochs, patience20',epochs=epoch,
        train=regression_scores(y[tr],mp[tr]),test=regression_scores(y[te],mp[te]))
    threshold=float(np.quantile(y[tr],1-q));labels=y>threshold
    cp,_,ce=fit(labels.astype(float),True)
    result['success_classifier']=dict(threshold=threshold,definition=f'improvement > analysis-train {(1-q)*100:g}th percentile; ties excluded',epochs=ce,
        test=classification_scores(labels[te],cp[te]),constant=classification_scores(labels[te],np.full(te.sum(),labels[tr].mean())))
    # Modest validation-screened candidate set; no exhaustive importance fishing.
    candidates=np.argsort(-abs(coef))[:5];rng=np.random.default_rng(seed);importance={}
    for j in candidates:
        effects=[]
        for _ in range(5):
            check();xp=xt[te].clone();xp[:,j]=xp[torch.tensor(rng.permutation(int(te.sum()))),j]
            with torch.no_grad():pp=net(xp).squeeze(-1).numpy()*scale+center
            effects.append(float(np.mean((pp-y[te])**2)-np.mean((mp[te]-y[te])**2)))
        importance[keys[j]]=dict(mean_mse_increase=float(np.mean(effects)),std=float(np.std(effects)),repeats=5)
    result['mlp_permutation_importance']=importance
    return result


def test_interactions(train,test,keys):
    result=[]
    for a,b in itertools.combinations(keys[:4],2):
        valid=[r for r in train if finite(r.get(a)) and finite(r.get(b))]
        if len(valid)<30:continue
        edges=[np.unique(np.quantile([r[k] for r in valid],[0,.25,.5,.75,1])) for k in [a,b]]
        cells=[]
        for i,j in itertools.product(range(len(edges[0])-1),range(len(edges[1])-1)):
            ys=[r['improvement'] for r in test if finite(r.get(a)) and finite(r.get(b)) and
                np.searchsorted(edges[0][1:-1],r[a],side='right')==i and np.searchsorted(edges[1][1:-1],r[b],side='right')==j]
            if len(ys)>=10:cells.append(dict(x=i,y=j,n=len(ys),mean=float(np.mean(ys)),std=float(np.std(ys))))
        result.append(dict(features=[a,b],edges=[e.tolist() for e in edges],cells=cells,definition='top4 analysis-train features; train quartile edges; test means/std; minimum10; exploratory'))
    return result


def calibration_analysis(rows,c,info,check):
    keys=blind_keys();result=dict(schema_version=1,experts={},reveal={},blind_features=keys,
        caveat='Pre-expert includes ground-truth baseline NLL/probability/margin. These are offline diagnostics, NOT available before next-token inference. Separate target-free probe provided. No task IDs, assigned weights, history, expert internals or post-expert features in blind matrices.')
    for i,e in enumerate(sorted({r['expert_id'] for r in rows})):
        check();group=[r for r in rows if r['expert_id']==e];split=split_rows(group);train=[r for r,s in zip(group,split) if s<6];test=[r for r,s in zip(group,split) if s>=8]
        # Explicit dictionary projection prevents metadata from entering feature search.
        blind=[{k:r.get(k) for k in keys+['improvement','expert_loss','relative_improvement','successful','relative_advantage']} for r in train]
        rankings={}
        for outcome in ['improvement','expert_loss','relative_improvement','successful','relative_advantage']:
            aa=[]
            for k in keys:
                check();aa.append(association(blind,k,outcome,bootstrap=0))
            rankings[outcome]=sorted(aa,key=lambda a:abs(a.get('spearman') or 0),reverse=True)
        candidates=[a['feature'] for a in rankings['improvement'] if a['spearman'] is not None][:8]
        confirmation=[association(test,k,'improvement') for k in candidates]
        p=probe(group,keys,c.seed,c.success_quantile,check)
        target_free=probe(group,blind_keys(True),c.seed,c.success_quantile,check)
        result['experts'][e]=dict(rankings=rankings,held_out_associations=confirmation,probes=p,target_free_probes=target_free,
            interactions=test_interactions(train,test,candidates),n=len(group),
            strength_associations=[dict(feature=k,spearman=association(test,k,'improvement',bootstrap=0)['spearman'],n=sum(finite(r.get(k)) for r in test)) for k in keys])
        # Reveal starts only after the blind model/feature choices above are fixed.
        table=info['sanity'][e];assigned=info['distributions'][i]['assigned'];avg=np.mean(list(assigned.values()))
        high=[t for t,w in assigned.items() if w>avg];low=[t for t,w in assigned.items() if w<=avg]
        gap=None;interval=None;differential=None
        if high and low:
            gap=float(np.mean([table[t]['mean_improvement'] for t in high])-np.mean([table[t]['mean_improvement'] for t in low]))
            others=[k for k in info['sanity'] if k!=e]
            differential=gap-float(np.mean([np.mean([info['sanity'][o][t]['mean_improvement'] for t in high])-np.mean([info['sanity'][o][t]['mean_improvement'] for t in low]) for o in others]))
            # Paired bootstrap same locations across experts, equal weighting across tasks.
            rng=np.random.default_rng(c.seed);diffs=[]
            for _ in range(300):
                means={}
                for t in table:
                    v=np.array(table[t]['improvements'])-np.mean([info['sanity'][o][t]['improvements'] for o in others],axis=0)
                    means[t]=float(rng.choice(v,len(v),replace=True).mean())
                diffs.append(np.mean([means[t] for t in high])-np.mean([means[t] for t in low]))
            interval=np.quantile(diffs,[.025,.975]).tolist()
        formed=gap is not None and gap>0 and differential>c.tie_epsilon and interval[0]>0
        # No-s0 comparison means NO strong/confirmed claim, regardless of a good R².
        score=p.get('mlp',{}).get('test',{}).get('r2');free_score=target_free.get('mlp',{}).get('test',{}).get('r2')
        confirmed=sum(a.get('interval') is not None and a['interval'][0]*a['interval'][1]>0 for a in confirmation)
        verdict='SPECIALIZATION DID NOT FORM' if not formed else ('DETECTABLE' if score is not None and score>.1 and confirmed>=2 else 'WEAKLY DETECTABLE' if score is not None and score>0 else 'NOT DETECTABLE')
        if c.specialization_strength==0:verdict='SPECIALIZATION DID NOT FORM' if not formed else 'WEAKLY DETECTABLE'
        breakdown={}
        for k in candidates[:4]:
            vals=[r[k] for r in train if finite(r.get(k))]
            threshold=float(np.quantile(vals,.75));by_task={}
            for task in table:
                rs=[r for r in test if r['task_id']==task and finite(r.get(k)) and r[k]>=threshold]
                by_task[task]=dict(n=len(rs),mean_improvement=float(np.mean([r['improvement'] for r in rs])) if len(rs)>=10 else None,assigned_weight=assigned[task])
            breakdown[k]=dict(high_threshold=threshold,tasks=by_task,warning='Descriptive reveal; not proof of cross-task transfer; tiny bins suppressed')
        result['reveal'][e]=dict(task_matrix={t:{k:v for k,v in d.items() if k!='improvements'} for t,d in table.items()},
            specialization_gap=gap,differential_gap=differential,differential_gap_ci=interval,specialization_formed=formed,
            blind_signal_task_breakdown=breakdown,verdict=verdict,verdict_provisional=True,target_free_r2=free_score,
            warning='Requires matched s=0 and seed replication. Formation requires positive assigned-minus-other improvement AND positive paired differential gap CI versus other experts; thresholds exploratory.',
            conclusion='KNOWN SPECIALIZATION EXISTS, BUT CURRENT TOKEN LAB INPUT DESCRIPTORS FAIL TO DETECT IT with these probes.' if formed and verdict=='NOT DETECTABLE' else verdict)
    return result


def calibration_report(c,info,a):
    lines=['# Forced Specialization Calibration','','## Experimental controls',
        f"Common source: {info['common_source']}; common hash: {info['common_hash']}",
        (f"Distinct architecture states recorded; shared unchanged: {info['shared_unchanged']}; strength: {c.specialization_strength}" if info.get('architecture_confounded') else f"Identical clones verified: {info['clone_identity_verified']}; shared unchanged: {info['shared_unchanged']}; strength: {c.specialization_strength}"),
        'Only expert module parameters train during specialization. Shared embeddings, position, context block, normalization and output head are frozen. Application is always h + expert(h).',
        'Budgets: '+json.dumps([{k:v for k,v in b.items() if k in ['expert_id','steps','samples','target_tokens','context_tokens']} for b in info['budgets']]), 'Optimizer: '+json.dumps(info['optimizer']),info['answer_protocol'],
        ('ARCHITECTURE-CONFOUNDED: common pretraining averages all expert endpoint losses on the same examples. 0% means equal general exposure, not identical experts or absent specialization.' if info.get('architecture_confounded') else 'Common pretraining optimizes one expert endpoint only (no separate baseline loss); all shared parameters then freeze.'),
        '', '## Blind signal discovery',a['caveat']]
    for e,d in a['experts'].items():
        lines+=['',f'### {e}', 'Features screened on analysis-train only; following effects measured on untouched analysis-test:']
        for v in d['held_out_associations']:lines.append(f"- {v['feature']}: Spearman {v['spearman']}, CI {v['interval']}, n={v['n']}")
        for name,p in [('All pre-expert (includes target-dependent)',d['probes']),('Target-free',d['target_free_probes'])]:
            lines.append(name)
            if p.get('status')!='ok':lines.append(p['status']);continue
            lines += ['Ridge test: '+json.dumps(p['ridge']['test']),'MLP test: '+json.dumps(p['mlp']['test']),
                      'Mean baseline: '+json.dumps(p['mean_baseline']),'Baseline NLL-only: '+json.dumps(p['baseline_loss_only']),
                      'Success classifier: '+json.dumps(p['success_classifier'])]
        lines+=['Held-out feature-pair cells: '+json.dumps(d['interactions'])]
    lines+=['','## No-specialization control','Compare a matched s=0 run in Strength Sweep. Without that control all verdicts are provisional; no STRONGLY DETECTABLE claim is made.',
            '', '## Reveal calibration labels','The following task labels were not available to blind feature selection or predictors.',
            'Assigned and actual sampling probabilities: '+json.dumps(info['distributions']), 'Realized task counts: '+json.dumps(info['budgets'])]
    for e,d in a['reveal'].items():
        lines += ['',f'### {e}', '| Task | n | Loss | Improvement | Median improvement | Win rate |','|---|---:|---:|---:|---:|---:|']
        for t,v in d['task_matrix'].items():lines.append(f"| {t} | {v['n']} | {v['mean_loss']:.4f} | {v['mean_improvement']:.4f} | {v['median_improvement']:.4f} | {v['win_rate']:.3f} |")
        lines += [f"Assigned-task gap: {d['specialization_gap']}; differential gap: {d['differential_gap']}; paired 95% CI: {d['differential_gap_ci']}",
                  'Blind high-feature regions by task (includes unassigned tasks; descriptive, not validated transfer): '+json.dumps(d['blind_signal_task_breakdown']),
                  'Provisional detection verdict: '+d['conclusion'],d['warning']]
    lines += ['', '## Specialization-strength sweep',
        'Matched completed runs available when this report was generated. The UI refreshes this comparison as more runs finish.',
        '| Run | Expert | Strength | Task gap | Differential gap | Blind MLP R² | Target-free R² | Success AUC |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in a.get('strength_sweep',[]):
        for e,v in r['experts'].items():lines.append(f"| {r['run_id']} | {e} | {r['strength']} | {v['gap']} | {v['differential_gap']} | {v['mlp_r2']} | {v['target_free_r2']} | {v['auc']} |")
    lines+=['','## Limitations','Task syntax and answer-prefix truncation may be proxies. Conditioning on fixed context excludes short examples. Generated examples can share templates. Baseline loss and improvement share a mathematical term; improvement prediction alone does not establish specialist identity. No descriptor can be declared information-free from failure of these finite-sample probes.']
    return '\n'.join(lines)


def sweep_record(c,info,a):
    from dataclasses import asdict
    cfg=asdict(c)
    # Exclude strength/source paths only; retain all analysis/training settings.
    ignore={'name','specialization_strength','common_base_source','common_checkpoint','specialist_checkpoint','save_raw'}
    fingerprint=hashlib.sha256(json.dumps({k:v for k,v in cfg.items() if k not in ignore},sort_keys=True).encode()+info['common_hash'].encode()).hexdigest()
    experts={}
    for e,d in a['experts'].items():
        p=d['probes'];tf=d['target_free_probes'];r=a['reveal'][e]
        experts[e]=dict(gap=r['specialization_gap'],differential_gap=r['differential_gap'],formed=r['specialization_formed'],verdict=r['verdict'],
            ridge_r2=p.get('ridge',{}).get('test',{}).get('r2'),mlp_r2=p.get('mlp',{}).get('test',{}).get('r2'),
            target_free_r2=tf.get('mlp',{}).get('test',{}).get('r2'),auc=p.get('success_classifier',{}).get('test',{}).get('roc_auc'),
            associations=d['strength_associations'])
    return dict(strength=c.specialization_strength,base_hash=info['common_hash'],comparison_key=fingerprint,experts=experts)
