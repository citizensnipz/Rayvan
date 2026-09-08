"""Exploratory analysis only. No feedback path into expert training."""
import hashlib
import itertools
import numpy as np
from .token_lab_features import SCHEMA


def ranks(x):
    _,inv,count=np.unique(x,return_inverse=True,return_counts=True)
    return (np.cumsum(count)-(count+1)/2)[inv]


def corr(x,y):
    if len(x)<3 or np.std(x)<1e-12 or np.std(y)<1e-12:return None
    return float(np.corrcoef(x,y)[0,1])


def finite(v):return bool(isinstance(v,(int,float)) and np.isfinite(v))


def quantiles(x,bins=5):
    edges=np.unique(np.quantile(x,np.linspace(0,1,bins+1)))
    return np.searchsorted(edges[1:-1],x,side='right'),edges


def association(rows,key,outcome,bootstrap=100):
    pairs=[(r.get(key),r.get(outcome)) for r in rows if finite(r.get(key)) and finite(r.get(outcome))]
    if len(pairs)<8:return dict(feature=key,n=len(pairs),pearson=None,spearman=None,interval=None,bins=[],mi=None)
    x,y=np.array(pairs).T;rx,ry=ranks(x),ranks(y);rho=corr(rx,ry);rng=np.random.default_rng(731)
    cis=[]
    for _ in range(bootstrap):
        i=rng.integers(len(x),size=len(x));v=corr(ranks(x[i]),ranks(y[i]))
        if v is not None:cis.append(v)
    bx,edges=quantiles(x);by,_=quantiles(y);joint=np.zeros((5,5))
    for a,b in zip(bx,by):joint[a,b]+=1
    joint/=len(x);den=joint.sum(1)[:,None]*joint.sum(0)[None,:];mask=joint>0
    mi=float((joint[mask]*np.log(joint[mask]/den[mask])).sum())
    bins=[dict(n=int((bx==i).sum()),x=float(x[bx==i].mean()),mean=float(y[bx==i].mean()),median=float(np.median(y[bx==i])),tiny=bool((bx==i).sum()<10)) for i in np.unique(bx)]
    between=sum(b['n']*(b['mean']-y.mean())**2 for b in bins)
    return dict(feature=key,n=len(x),pearson=corr(x,y),spearman=rho,interval=np.quantile(cis,[.025,.975]).tolist() if len(cis)>20 else None,
        bins=bins,mi=mi,binned_eta_squared=float(between/np.square(y-y.mean()).sum()) if np.var(y)>1e-12 else None,
        direction='positive' if rho is not None and rho>0 else 'negative' if rho is not None and rho<0 else 'unavailable')


def low_high(rows,keys,outcome='improvement',q=.2):
    valid=[r for r in rows if finite(r.get(outcome))]
    if len(valid)<10:return []
    bounds=np.quantile([r[outcome] for r in valid],[q,1-q]);out=[]
    for key in keys:
        groups=[[r[key] for r in valid if finite(r.get(key)) and cond(r[outcome])] for cond in [lambda y:y<=bounds[0],lambda y:bounds[0]<y<bounds[1],lambda y:y>=bounds[1]]]
        def stat(a):return dict(n=len(a),mean=float(np.mean(a)),median=float(np.median(a)),std=float(np.std(a))) if a else dict(n=0,mean=None,median=None,std=None)
        low,mid,high=map(stat,groups);den=np.sqrt((np.var(groups[0])+np.var(groups[2]))/2) if groups[0] and groups[2] else 0
        out.append(dict(feature=key,low=low,middle=mid,high=high,effect=float((high['mean']-low['mean'])/den) if den>1e-12 else None))
    return out


def multivariate(rows,keys,outcome):
    # Group assignments are identical across experts/pairs. No target-derived or response features.
    rows=[r for r in rows if finite(r.get(outcome))]
    if len(rows)<30:return dict(status='insufficient samples',n=len(rows))
    train=np.array([int(hashlib.sha256(str(r['sample_id']).encode()).hexdigest()[:8],16)%10<7 for r in rows])
    if train.sum()<15 or (~train).sum()<8:return dict(status='insufficient split sizes')
    keys=[k for k in keys if sum(finite(r.get(k)) for r,t in zip(rows,train) if t)>=max(10,train.sum()*.8)]
    if not keys:return dict(status='no sufficiently available input features')
    X=np.array([[r.get(k) if finite(r.get(k)) else np.nan for k in keys] for r in rows]);y=np.array([r[outcome] for r in rows])
    median=np.nanmedian(X[train],axis=0);X=np.where(np.isnan(X),median,X)
    mean=X[train].mean(0);std=X[train].std(0);keep=std>1e-8;keys=[k for k,v in zip(keys,keep) if v]
    if not keys:return dict(status='all inputs constant')
    X=(X[:,keep]-mean[keep])/std[keep];center=y[train].mean();alpha=10.
    coef=np.linalg.solve(X[train].T@X[train]+alpha*np.eye(len(keys)),X[train].T@(y[train]-center));pred=X@coef+center
    def score(mask):
        err=pred[mask]-y[mask];den=np.square(y[mask]-y[mask].mean()).sum()
        return dict(n=int(mask.sum()),r2=float(1-np.square(err).sum()/den) if den>1e-12 else None,mae=float(abs(err).mean()),constant_mae=float(abs(y[mask]-center).mean()))
    return dict(status='ok',train=score(train),held_out=score(~train),ridge_alpha=alpha,
        coefficients=dict(zip(keys,map(float,coef))),split='sample-id SHA256 70/30; preprocessing fits analysis-train only',
        inputs='pre-expert features only; no target NLL, gradient, response, rank or loss-derived features')


def heatmap(rows,a,b,outcome):
    rows=[r for r in rows if all(finite(r.get(k)) for k in [a,b,outcome])]
    if len(rows)<30:return dict(features=[a,b],cells=[])
    x=np.array([r[a] for r in rows]);y=np.array([r[b] for r in rows]);z=np.array([r[outcome] for r in rows]);ix,ex=quantiles(x,4);iy,ey=quantiles(y,4)
    cells=[]
    for i,j in itertools.product(range(4),repeat=2):
        mask=(ix==i)&(iy==j);n=int(mask.sum())
        if n>=10:cells.append(dict(x=i,y=j,n=n,mean=float(z[mask].mean())))
    return dict(features=[a,b],x_edges=ex.tolist(),y_edges=ey.tolist(),cells=cells,minimum_cell_n=10)


def analyze(rows,seed=42,cancelled=lambda:False):
    keys=[k for k,v in SCHEMA.items() if v['category']!='Outcome'];input_keys=[k for k in keys if SCHEMA[k]['role']=='input']
    result=dict(experts={},pairs={},schema_version=1,caveat='Exploratory; no multiplicity-adjusted claims. Bootstrap intervals are pointwise. MI/eta-squared are in-sample and positively biased. Repeated related texts may remain dependent. Ridge failure does not rule out nonlinear information.')
    for expert in sorted({r['expert_id'] for r in rows}):
        if cancelled():raise InterruptedError('cancelled during analysis')
        group=[r for r in rows if r['expert_id']==expert];assocs={}
        for outcome in ['expert_loss','improvement','relative_advantage']:
            values=[]
            for k in keys:
                if cancelled():raise InterruptedError('cancelled during feature analysis')
                values.append(association(group,k,outcome))
            assocs[outcome]=sorted(values,key=lambda a:abs(a.get('spearman') or 0),reverse=True)
        performance={k:float(np.mean([r[k] for r in group if finite(r.get(k))])) if any(finite(r.get(k)) for r in group) else None for k in ['expert_loss','improvement','is_best','tied','best_second_margin']}
        performance.update(n=len(group),median_loss=float(np.median([r['expert_loss'] for r in group])))
        top=[a['feature'] for a in assocs['improvement'] if a['feature'] in input_keys and a.get('spearman') is not None][:4]
        redundancy=[]
        for a,b in itertools.combinations(keys,2):
            if cancelled():raise InterruptedError('cancelled during redundancy analysis')
            pair=[(r.get(a),r.get(b)) for r in group if finite(r.get(a)) and finite(r.get(b))]
            if len(pair)<10:continue
            x,y=np.array(pair).T;v=corr(x,y);s=corr(ranks(x),ranks(y))
            redundancy.append(dict(a=a,b=b,pearson=v,spearman=s,n=len(pair)))
        result['experts'][expert]=dict(performance=performance,associations=assocs,low_high=low_high(group,keys),
            ridge={o:multivariate(group,input_keys,o) for o in ['improvement','relative_advantage']},
            interactions=[heatmap(group,a,b,'improvement') for a,b in itertools.combinations(top,2)],correlations=redundancy)
    # Same-state, independent pairs only. Features are from A; response attributes not admissible.
    ids=sorted(result['experts'])
    for a,b in itertools.combinations(ids,2):
        left={r['sample_id']:r for r in rows if r['expert_id']==a and r['topology']=='independent'}
        paired=[]
        for r in rows:
            if r['expert_id']==b and r['sample_id'] in left:
                l=left[r['sample_id']];paired.append(dict(l,pair_loss_difference=l['expert_loss']-r['expert_loss']))
        if not paired:continue
        associations=sorted([association(paired,k,'pair_loss_difference') for k in input_keys],key=lambda a:abs(a.get('spearman') or 0),reverse=True)
        eps=paired[0]['tie_epsilon'];groups={name:[r for r in paired if cond(r['pair_loss_difference'])] for name,cond in [('A wins',lambda x:x < -eps),('tie',lambda x:abs(x)<=eps),('B wins',lambda x:x>eps)]}
        candidates=[a['feature'] for a in associations if a.get('spearman') is not None][:4]
        result['pairs'][f'{a} vs {b}']=dict(n=len(paired),sign='negative means A better',associations=associations,
            win_groups={name:dict(n=len(g),features={k:dict(n=sum(finite(r.get(k)) for r in g),mean=float(np.mean([r[k] for r in g if finite(r.get(k))])) if any(finite(r.get(k)) for r in g) else None) for k in input_keys}) for name,g in groups.items()},
            ridge=multivariate(paired,input_keys,'pair_loss_difference'),interactions=[heatmap(paired,a,b,'pair_loss_difference') for a,b in itertools.combinations(candidates,2)])
    return result


def report(config,summary,analysis):
    def fmt(v):return 'unavailable' if v is None else f'{v:.4f}'
    def describe(a,outcome):
        means=[b['mean'] for b in a.get('bins',[]) if not b['tiny']]
        diffs=np.diff(means)
        shape='insufficient adequately populated bins' if len(means)<3 else ('bin means consistently ordered' if (diffs>=0).all() or (diffs<=0).all() else 'bin means are not consistently ordered')
        return f"- **{a['feature']}**: higher values coincide with {'higher' if a['spearman']>0 else 'lower'} {outcome}; Pearson {fmt(a['pearson'])}, Spearman {fmt(a['spearman'])}, pointwise 95% bootstrap {a['interval']}, n={a['n']}; {shape}. Role: {SCHEMA[a['feature']]['role']}. Association, not causation."
    def probe_lines(probe):
        if probe.get('status')!='ok':return ['INCONCLUSIVE: '+probe.get('status','unavailable')]
        t,h=probe['train'],probe['held_out']
        return [f"Analysis-train n={t['n']}, R²={fmt(t['r2'])}; held-out n={h['n']}, R²={fmt(h['r2'])}, MAE={fmt(h['mae'])}; training-mean constant MAE={fmt(h['constant_mae'])}.",
                'Standardized coefficients (diagnostic, not causal): '+', '.join(f"{k}={v:.4f}" for k,v in sorted(probe['coefficients'].items(),key=lambda kv:abs(kv[1]),reverse=True))]
    def regions(items,outcome):
        output=[]
        for item in items:
            for cell in item['cells']:
                output.append(f"- {item['features'][0]} bin {cell['x']+1} × {item['features'][1]} bin {cell['y']+1}: mean {outcome}={cell['mean']:.4f}, n={cell['n']}.")
        return output or ['No feature-pair regions with sufficient observations.']
    lines=['# Token Lab Report','', '## Experiment',
        f"Dataset: {config['dataset']}; topology: {config['topology']}; experts: {', '.join(config['families'])}; seed: {config['seed']}.",
        f"Training: {config['train_steps']} steps × {config['batch_size']} endpoint targets. Maximum context {config['sequence_length']}; causal neighbourhood {config['window']}. Spectral: {config['spectral_enabled']} (rate {config['spectral_rate']}); deep: {config['deep_enabled']} (rate {config['deep_rate']}).",
        f"Measured {summary['observations']} expert rows on {summary['measured_locations']} held-out locations. Analysis includes {summary['analysis_locations']} uniformly reservoir-sampled locations (all locations when below cap).",
        '', 'Difficulty is not specialization: raw loss associations can be shared by all experts. Same-state advantage is the specialization outcome. Serial stage effects are not counterfactual advantages.', '',analysis['caveat'],
        '', 'Full-stream mean performance (before any analysis sampling):']
    for row in summary['overall_full_stream']:
        lines.append(f"- {row['expert_id']}: n={row['n']}, loss={fmt(row['mean_loss'])}, improvement={fmt(row['mean_improvement'])}.")
    for name,data in analysis['experts'].items():
        p=data['performance']
        lines+=['',f'## Overall performance — {name} (analysis sample)',
            f"n={p['n']}; mean loss={fmt(p['expert_loss'])}; median loss={fmt(p['median_loss'])}; mean improvement={fmt(p['improvement'])}; within-tolerance win rate={fmt(p['is_best'])}; tie rate={fmt(p['tied'])}; mean best–second margin={fmt(p['best_second_margin'])}."]
        for title,outcome,sign in [('LOWER loss','expert_loss',-1),('HIGHER loss','expert_loss',1),('GREATER improvement','improvement',1),('WORSE improvement','improvement',-1)]:
            lines+=['',f'### Attributes associated with {title}']
            candidates=[a for a in data['associations'][outcome] if a.get('spearman') is not None and a['spearman']*sign>0][:8]
            lines += [describe(a,outcome) for a in candidates] or ['INCONCLUSIVE: insufficient nonconstant observations.']
        outcome='relative_advantage' if config['topology']=='independent' and len(config['families'])>1 else 'improvement'
        nulls=[a['feature'] for a in data['associations'][outcome] if a.get('spearman') is not None and abs(a['spearman'])<.1]
        unavailable=[a['feature'] for a in data['associations']['improvement'] if a.get('spearman') is None]
        lines+=['','### Features with little observed univariate relationship',
            'Small descriptive monotonic effects on '+outcome+' (|rho|<0.1; not proof of absence): '+(', '.join(nulls) or 'none meeting this descriptive threshold'),
            'Unavailable/degenerate or insufficient samples: '+(', '.join(unavailable) or 'none')]
        nonlinear=sorted([a for a in data['associations']['improvement'] if a.get('pearson') is not None and abs(a['pearson'])<.1 and (a.get('binned_eta_squared') or 0)>.1],key=lambda a:a['binned_eta_squared'],reverse=True)
        lines+=['','### Strongest nonlinear candidates']
        lines += [f"- {a['feature']}: n={a['n']}, quantile MI={fmt(a['mi'])} nats, in-sample binned eta²={fmt(a['binned_eta_squared'])}." for a in nonlinear[:8]] or ['No candidates meeting the exploratory screening thresholds.']
        lines+=['Discretized quantile MI and eta-squared are biased exploratory scores, not validated nonlinear predictions.',
            '','### Strongest feature pairs']+regions(data['interactions'],'improvement')
        lines+=['Feature pairs were selected on these same observations; confirm separately. Bin edges are in analysis.json.',
            '','### Redundant features']
        red=[a for a in data['correlations'] if abs(a.get('spearman') or 0)>.95]
        lines += [f"- {a['a']} / {a['b']}: Spearman {fmt(a['spearman'])}, n={a['n']}." for a in red[:25]] or ['No pairs above |rho|=0.95.']
        lines+=['','### Multivariate predictability']
        for o,probe in data['ridge'].items():lines+=['',o]+probe_lines(probe)
    lines+=['','## Expert-specific advantage']
    for name,data in analysis['pairs'].items():
        lines += ['',name,f"n={data['n']}; negative L_A-L_B favours A. "+'; '.join(f"{k}: n={v['n']}" for k,v in data['win_groups'].items())]
        lines += [describe(a,'L_A-L_B') for a in data['associations'][:5] if a.get('spearman') is not None]
        lines += probe_lines(data['ridge'])+regions(data['interactions'],'L_A-L_B')
    if not analysis['pairs']:lines.append('Unavailable: serial stages or a single expert do not permit same-state expert advantage.')
    lines+=['','## Conclusions']
    for name,data in analysis['experts'].items():
        for outcome,probe in data['ridge'].items():
            if probe.get('status')!='ok':lines.append(f"INCONCLUSIVE: {name} / {outcome}: {probe.get('status')}.");continue
            score=probe['held_out'];r2=score['r2']
            verdict='WEAKLY SUPPORTS' if r2 is not None and r2>.1 and score['mae']<score['constant_mae'] else 'FAILS TO SUPPORT'
            lines.append(f"{verdict} useful linear predictability of {name} {outcome} from recorded pre-expert features in this split: held-out R²={fmt(r2)}, MAE={fmt(score['mae'])}, constant MAE={fmt(score['constant_mae'])}, n={score['n']}. This does not establish presence or absence of nonlinear information.")
    lines+=['INCONCLUSIVE about causal mechanisms. Pre-expert input predictability must be distinguished from target-derived and post-response associations. Confirm patterns on a new seed/dataset before designing a router. No architectural recommendation follows automatically.']
    return '\n'.join(lines)+'\n'
