"""Exploratory developmental evidence; evaluation never feeds expert assignment."""
import itertools
import json
import math
import numpy as np
from .token_lab import write
from .token_lab_analysis import corr,ranks
from .token_lab_calibration_analysis import probe
from .token_lab_discovery import paired_interval
from .token_lab_features import blind_keys


def safe_corr(a,b):return corr(np.asarray(a),np.asarray(b))


def performance_signature(loss,baseline,epsilon):
    I=baseline[:,None]-loss;adv=(I.sum(1)[:,None]-I)/(I.shape[1]-1);adv=I-adv
    interaction=I-I.mean(0)-I.mean(1)[:,None]+I.mean()
    ties=loss<=loss.min(1)[:,None]+epsilon;shares=ties/ties.sum(1)[:,None]
    unique=ties.sum(1)==1;wins=np.bincount(loss.argmin(1)[unique],minlength=loss.shape[1])
    return dict(interaction_rms=float(np.sqrt(np.mean(interaction**2))),
        oracle_gain_over_best_constant=float(loss.mean(0).min()-loss.min(1).mean()),
        tie_rate=float(np.mean(ties.sum(1)>1)),unique_winner_counts=wins.tolist(),
        meaningful_winning_experts=int(np.sum(wins>=max(2,.05*len(loss)))),
        fractional_win_rates=shares.mean(0).tolist(),mean_improvements=I.mean(0).tolist(),
        improvement_variance=I.var(0).tolist(),mean_population_advantage=adv.mean(0).tolist(),
        improvement_correlations=[dict(a=i,b=j,correlation=safe_corr(I[:,i],I[:,j])) for i,j in itertools.combinations(range(loss.shape[1]),2)]),I,adv


def temporal(previous,I,adv,scores,loss,q):
    if previous is None:return dict(status='initial checkpoint')
    per=[];k=max(1,int(len(I)*q))
    for e in range(I.shape[1]):
        a=set(np.argsort(previous['improvement'][:,e])[-k:]);b=set(np.argsort(I[:,e])[-k:])
        per.append(dict(expert=e,top_success_jaccard=len(a&b)/len(a|b) if np.ptp(I[:,e])>1e-8 and np.ptp(previous['improvement'][:,e])>1e-8 else None,improvement_correlation=safe_corr(previous['improvement'][:,e],I[:,e]),advantage_correlation=safe_corr(previous['advantage'][:,e],adv[:,e]),preference_correlation=safe_corr(previous['scores'][:,e],scores[:,e])))
    stable=[v['advantage_correlation'] for v in per if v['advantage_correlation'] is not None]
    # Undefined constant signatures are not stable specialization.
    return dict(status='matched evaluation panel',experts=per,mean_advantage_correlation=float(np.mean(stable)) if stable else None,
        winner_persistence=float(np.mean(previous['loss'].argmin(1)==loss.argmin(1))),
        pairwise_rank_persistence=float(np.mean([np.mean((previous['loss'][:,i]<previous['loss'][:,j])==(loss[:,i]<loss[:,j])) for i,j in itertools.combinations(range(I.shape[1]),2)])))


def checkpoint_analysis(c,seed,step,pools,L,base,disagreement,D,db,router,fixed,previous,root,check,oracle_policy=False):
    E=L.shape[1];rows=pools['evaluation'];preferences=[router.scores(v['features']) for v in rows]
    scores=np.array([v[0] for v in preferences]);chosen=L.argmin(1) if oracle_policy else np.array([router.choose(v['features']) for v in rows])
    selected=L[np.arange(len(L)),chosen];fixedL=L[:,fixed];uniform=L.mean(1);oracle=L.min(1);rr=L[np.arange(len(L)),np.arange(len(L))%E]
    metadata=[v['meta'] for v in rows]
    utility=dict(policy_type='privileged_oracle' if oracle_policy else 'validation_selected_competence',policy_loss=float(selected.mean()),fixed=float(fixedL.mean()),fixed_expert=fixed,uniform=float(uniform.mean()),round_robin=float(rr.mean()),oracle=float(oracle.mean()),
        gain_over_fixed=float((fixedL-selected).mean()),gain_over_fixed_ci=paired_interval(metadata,fixedL-selected,seed),
        gain_over_random=float((uniform-selected).mean()),gain_over_random_ci=paired_interval(metadata,uniform-selected,seed),oracle_regret=float((selected-oracle).mean()),
        routing_usage=np.bincount(chosen,minlength=E).tolist(),preference_logit_max=max(v[1] for v in preferences),
        preference_margin_mean=float(np.mean(np.sort(scores,axis=1)[:,-1]-np.sort(scores,axis=1)[:,-2])))
    utility['gain_over_each_fixed']=[dict(expert=e,gain=float((L[:,e]-selected).mean()),ci=paired_interval(metadata,L[:,e]-selected,seed)) for e in range(E)]
    utility['meaningful_selected_experts']=int(np.sum(np.bincount(chosen,minlength=E)>=max(2,.05*len(L))))
    spec,I,adv=performance_signature(L,base,c['tie_epsilon']);spec['output_delta_disagreement_rms']=float(disagreement.mean())
    spec['router_preference_advantage_spearman']=[safe_corr(ranks(scores[:,e]),ranks(adv[:,e])) for e in range(E)]
    utility['best_expert_agreement']=float(np.mean(L[np.arange(len(L)),chosen]<=L.min(1)+c['tie_epsilon']))
    stability=temporal(previous,I,adv,scores,L,c['success_quantile'])
    with (root/'observations.jsonl').open('w',encoding='utf-8') as f:
        for i,v in enumerate(rows):
            for e in range(E):
                rank=1+int(np.sum(L[i]<L[i,e]-c['tie_epsilon']))
                f.write(json.dumps(dict(v['meta']|v['features'],expert_id=str(e),expert_family=c['family'],expert_stage=1,checkpoint_step=step,
                    baseline_loss=float(base[i]),expert_loss=float(L[i,e]),improvement=float(I[i,e]),relative_advantage=float(adv[i,e]),expert_rank=rank,
                    chosen_expert=int(chosen[i]),preference=float(scores[i,e]),best_expert=int(L[i].argmin())))+'\n')
    diagnostics={}
    for e in range(E):
        check();dr=[dict(**v['features'],sample_id=v['meta']['sample_id'],analysis_group=v['meta']['analysis_group'],baseline_loss=float(b),improvement=float(b-l[e])) for v,b,l in zip(pools['diagnostic'],db,D)]
        diagnostics[str(e)]=probe(dr,blind_keys(True),seed,c['success_quantile'],check)
    # Reveal only after blind inputs/probe choices are fixed; labels never enter router.
    reveal={}
    for task in sorted({v['meta']['task_id'] for v in rows}):
        mask=np.array([v['meta']['task_id']==task for v in rows])
        reveal[task]=dict(n=int(mask.sum()),expert_mean_loss=L[mask].mean(0).tolist(),expert_mean_improvement=I[mask].mean(0).tolist(),routed_loss=float(selected[mask].mean()))
    write(root/'diagnostic-observations.json',dict(features=[v['features'] for v in pools['diagnostic']],metadata=[v['meta'] for v in pools['diagnostic']],baseline=db.tolist(),losses=D.tolist()))
    warnings=[]
    if max(utility['routing_usage'])/len(L)>.95:warnings.append('Inference concentration above 95%; do not interpret as specialization')
    if utility['preference_logit_max']>20 and utility['gain_over_fixed']<=0:warnings.append('Large preference logits without held-out selection benefit')
    if spec['interaction_rms']<c['tie_epsilon']:warnings.append('Expert competence functions effectively indistinguishable at configured tolerance')
    if previous is not None and selected.mean()>previous['routed_loss']+c['tie_epsilon']:warnings.append('Routed evaluation loss worsened since preceding checkpoint')
    if len(L)<100:warnings.append('Small evaluation panel; smoke-level evidence only')
    return dict(specialization=spec,stability=stability,utility=utility,detectability=diagnostics,reveal=reveal,warnings=warnings),dict(improvement=I,advantage=adv,scores=scores,loss=L,routed_loss=float(selected.mean()))


def stats(values,seed=42):
    v=np.asarray(values,dtype=float)
    if not len(v):return dict(n=0,mean=None,median=None,std=None,ci=None,values=[])
    rng=np.random.default_rng(seed);ci=np.quantile([rng.choice(v,len(v),replace=True).mean() for _ in range(1000)],[.025,.975]).tolist() if len(v)>1 else None
    return dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),std=float(v.std(ddof=1)) if len(v)>1 else None,ci=ci,values=v.tolist(),note='Seed bootstrap, exploratory; three seeds give imprecise uncertainty')


def summarize(entries,c):
    good=[e for e in entries if e['status']=='completed'];byid={};comparisons=[]
    for e in good:byid.setdefault(e['id'],[]).append(e)
    def match(e,label,a,b):return next((v for v in good if v['seed']==e['seed'] and v['condition']==label and v['asymmetry']==a and v['feedback']==b),None)
    for e in good:
        specs=[]
        if e['condition']=='B':specs=[('B - A','A',0.,0.)]
        if e['condition']=='C':specs=[('C - A','A',0.,0.)]
        if e['condition']=='D':specs=[('D - B','B',e['asymmetry'],0.),('D - E','E',e['asymmetry'],e['feedback'])]
        if e['condition']=='F':specs=[('F - D','D',e['asymmetry'],e['feedback'])]
        for label,other,a,b in specs:
            v=match(e,other,a,b)
            if v is None:continue
            lhs=e['final']['utility']['oracle' if e['condition']=='F' else 'policy_loss'];rhs=v['final']['utility']['policy_loss']
            comparisons.append(dict(comparison=label,asymmetry=e['asymmetry'],feedback=e['feedback'],seed=e['seed'],loss_difference=lhs-rhs,loss_gain=rhs-lhs,
                interaction_difference=e['final']['specialization']['interaction_rms']-v['final']['specialization']['interaction_rms'],
                total_training_applications_equal=e['cost']['training_applications']==v['cost']['training_applications'],
                per_expert_exposure_equal=e['final']['exposure']==v['final']['exposure'],
                note='F uses privileged oracle evaluation; reference bound, not deployable advantage' if e['condition']=='F' else 'Both final populations evaluated with their own learned competence router'))
    grouped={}
    for v in comparisons:
        key=f"{v['comparison']} / a{v['asymmetry']:g} / b{v['feedback']:g}";grouped.setdefault(key,[]).append(v)
    paired={k:dict(loss_difference=stats([v['loss_difference'] for v in vs]),loss_gain=stats([v['loss_gain'] for v in vs]),interaction_difference=stats([v['interaction_difference'] for v in vs]),seeds=[v['seed'] for v in vs]) for k,vs in grouped.items()}
    conditions={}
    for key,es in byid.items():
        formed=[];detected=[];useful=[]
        for e in es:
            f=e['final'];first=e['checkpoints'][0];control=match(e,'A',0.,0.)
            stable=(f['stability'].get('mean_advantage_correlation') or 0)>.5
            formed.append(bool(e['asymmetry']>0 and control and f['specialization']['interaction_rms']>max(first['specialization']['interaction_rms'],control['final']['specialization']['interaction_rms'])+c['tie_epsilon'] and f['specialization']['meaningful_winning_experts']>=2 and stable))
            detected.append(e['condition']!='F' and all(v['ci'][0]>0 for v in f['utility']['gain_over_each_fixed']) and f['utility']['meaningful_selected_experts']>=2 and f['specialization']['meaningful_winning_experts']>=2)
            bs=[match(e,'B',e['asymmetry'],0.),match(e,'E',e['asymmetry'],e['feedback'])] if e['condition']=='D' else []
            useful.append(bool(len(bs)==2 and all(v and f['utility']['policy_loss']<v['final']['utility']['policy_loss'] for v in bs)))
        complete=len(es)==len(c['seeds']);n=len(es)
        def verdict(flags,applicable=True):
            if not applicable or not complete or n<3:return 'INCONCLUSIVE'
            if all(flags):return 'SUPPORTED' if n>=5 else 'WEAKLY SUPPORTED'
            if any(flags):return 'INCONCLUSIVE'
            return 'NOT SUPPORTED'
        conditions[key]=dict(seeds=[e['seed'] for e in es],final_loss=stats([e['final']['utility']['policy_loss'] for e in es]),
            final_uniform_loss=stats([e['final']['utility']['uniform'] for e in es]),interaction=stats([e['final']['specialization']['interaction_rms'] for e in es]),router_gain=stats([e['final']['utility']['gain_over_fixed'] for e in es]),
            best_retrospective_loss=stats([min(v['utility']['policy_loss'] for v in e['checkpoints']) for e in es]),
            hypothesis_1=verdict(formed,es[0]['asymmetry']>0),hypothesis_2=verdict(detected,es[0]['condition']!='F'),hypothesis_3=verdict(useful,es[0]['condition']=='D'),
            evidence_counts=dict(formed=sum(formed),detected=sum(detected),utility_over_B_and_E=sum(useful)),
            missing_seeds=[s for s in c['seeds'] if s not in {e['seed'] for e in es}])
    return dict(conditions=conditions,paired_comparisons=paired,individual_comparisons=comparisons,
        failures=[dict(id=e['id'],seed=e['seed'],status=e['status'],error=e.get('error')) for e in entries if e['status']!='completed'],
        verdict_policy='Exploratory thresholds: growth beyond initial/A interaction RMS by tie tolerance, >=2 nontrivial winners, adjacent advantage correlation >.5; detection requires positive gain interval against every constant expert and >=2 meaningfully selected experts; utility requires paired D lower than B and E. F and inapplicable controls are INCONCLUSIVE. <3 seeds or missing seeds INCONCLUSIVE, unanimous 3–4 seeds WEAKLY SUPPORTED, unanimous >=5 SUPPORTED; mixed seed evidence INCONCLUSIVE. Inspect intervals and trajectories, not verdict alone.')


def report(result):
    c=result['settings'];s=result['summary']
    lines=['# Emergent Specialization',f"Status: {result['status']}. Family: {c['family']}; experts: {c['expert_count']}; seeds: {c['seeds']}.",
        'Task-agnostic Gaussian parameter perturbations; same mixed inputs across conditions; shared encoder/readout frozen. One residual expert application per prediction. No task-assigned roles.',
        'Validated full/simple observer and pairwise regret-weighted logistic learner reused. For >2 experts, pairwise win-count aggregation is an unvalidated population extension, not a replacement representation or loss.',
        'A identical/equal; B perturbation/equal; C no perturbation/feedback; D perturbation/feedback; E D ticket counts randomly permuted across inputs within each minibatch; optional F privileged oracle feedback.',
        'Each input has E training tickets. A/B assign one to each expert. A fraction beta is redirected using competence, with mandatory epsilon uniform exploration. Total E*B applications per update block are fixed; optimizer update counts may differ and are recorded.',
        'Router refreshes use fresh losses on a fixed TRAINING probe panel. Validation chooses old/candidate/fixed router. Evaluation and diagnostic panels never influence training. Shared states and descriptor references are fixed and common to all conditions.',
        'Expert learning rate and optimizer settings are identical. No FLOP estimate is claimed; forward applications, token exposures, optimizer updates, probe/router/diagnostic time and wall time are separate.',
        '\n## Hypotheses and seed consistency','| Condition | Final policy NLL | Interaction RMS | Router gain | H1 emergence | H2 detection | H3 utility |','|---|---:|---:|---:|---|---|---|']
    for key,g in s['conditions'].items():lines.append(f"| {key} | {g['final_loss']['mean']:.5f} | {g['interaction']['mean']:.5f} | {g['router_gain']['mean']:.5f} | {g['hypothesis_1']} | {g['hypothesis_2']} | {g['hypothesis_3']} |")
    lines+=['\n'+s['verdict_policy'],'\n## Paired condition effects','Negative loss difference means the left-hand condition improved loss. Each pair uses the same seed/common base/evaluation panel.','| Comparison | Seeds | Mean loss difference | 95% seed interval | Interaction difference |','|---|---|---:|---|---:|']
    for key,g in s['paired_comparisons'].items():lines.append(f"| {key} | {g['seeds']} | {g['loss_difference']['mean']:.5f} | {g['loss_difference']['ci']} | {g['interaction_difference']['mean']:.5f} |")
    lines+=['\n## Individual seeds, trajectories and warnings']
    for e in result['entries']:
        lines.append(f"\n### Seed {e['seed']} {e['id']}: {e['status']}")
        if e['status']!='completed':lines.append(str(e.get('error','Incomplete')));continue
        lines.append('Evaluation policy: '+e['final']['utility']['policy_type'])
        lines.append('Realized perturbations: '+str([v['relative_delta_norm'] for v in e['initialization']['experts']]))
        for m in e['checkpoints']:
            lines.append(f"- Step {m['step']}: NLL {m['utility']['policy_loss']:.5f}; interaction {m['specialization']['interaction_rms']:.5f}; oracle opportunity {m['specialization']['oracle_gain_over_best_constant']:.5f}; router gain {m['utility']['gain_over_fixed']:.5f}; stability {m['stability'].get('mean_advantage_correlation')}; training applications {m['cost']['training_applications']}; exposures {m['exposure']}; warnings {m['warnings']}")
        lines.append('Cost: '+str(e['cost']))
        lines.append('Final held-out detectability per expert (input-only diagnostic predictors; scores are not the online router):')
        for expert,d in e['final']['detectability'].items():
            lines.append(f"- Expert {expert}: {d.get('status')}; ridge {d.get('ridge',{}).get('test')}; MLP {d.get('mlp',{}).get('test')}; success {d.get('success_classifier',{}).get('test')}; mean baseline {d.get('mean_baseline')}; target-aware difficulty-only diagnostic {d.get('baseline_loss_only')}")
        lines.append('Post-hoc task reveal (not used in the preceding analysis): '+str(e['final']['reveal']))
    lines+=['\n## Limits and negative evidence',
        'Parameter distance is not specialization. Improvement prediction can reflect common difficulty; consult state-by-expert interaction, distinct unique winners, relative-advantage stability and above-fixed routing gains.',
        'High concentration, zero exposure, stale/insufficient probes, identical functions and worsening loss remain visible. All-zero identity controls can predict common difficulty without demonstrating specialization.',
        'Best checkpoint loss is retrospective descriptive data, never a selection criterion. Repeated evaluation-panel curves are correlated. Confidence intervals are exploratory and not multiplicity-adjusted.',
        'Ordinary mixed training here means expert-only training under a frozen general representation. It does not establish emergence with a jointly changing trunk, multi-step EMC integration, heterogeneous expert architectures or unseen tasks.',
        'Per-expert optimizer updates and unique examples can differ despite matched total application budgets. E controls D marginal usage exactly within each block; random assignments may accidentally coincide with D.',
        'F is a privileged one-step reference, not a rigorous upper bound on future developmental utility. Source common bases are reused; with a single checkpoint, seeds do not replicate common pretraining.',
        'Raw observations and diagnostic inputs, router/checkpoint weights, initialization hashes, assignments and matched data are saved. Share emergence-analysis.json and this report.']
    return '\n'.join(lines)
