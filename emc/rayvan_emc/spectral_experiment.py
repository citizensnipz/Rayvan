"""Controlled fits on identical existing FitBank observations; no fit-time experts.

Run python -m rayvan_emc.spectral_experiment --bank router-fit-bank.pt --output DIR
or --smoke for a small synthetic-token integration experiment (not real evidence).
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from .model import EMCConfig
from .nexus import GeometricNexusRouter
from .spectral_routing import SpectralGeometricRouter, routing_objective, RunningStandardizer
from .spectral_geometry import Descriptor
from .spectral_config import spectral_config
from .value_fit import FitBank, measure_bank, load_bank_payload
from .value_routing import CounterfactualValueEMC, StateBatch
from .transformation_geometry import transformation_signature, complementarity, transformation_names


def metrics(scores, losses, train_means, tie_epsilon=1e-3, router_temperature=.25):
    rows = []
    for depth, (s, y, mean) in enumerate(zip(scores, losses, train_means)):
        selected, best = s.argmax(-1), y.argmin(-1)
        regret = y.gather(1, selected[:, None]).squeeze(1) - y.amin(-1)
        a, b = torch.triu_indices(y.size(1), y.size(1), 1)
        gap = y[:, a]-y[:, b]
        comparable = gap.abs() > tie_epsilon
        correct = ((s[:, a]-s[:, b]) * gap) < 0
        ordered = y.sort(-1).values
        margin = ordered[:, 1]-ordered[:, 0] if y.size(1)>1 else y[:, 0]*0
        rows.append(dict(depth=depth, routing_regret=float(regret.mean()),
          oracle_top1=float((selected==best).float().mean()),
          oracle_top2=float((s.topk(min(2,s.size(1))).indices==best[:,None]).any(-1).float().mean()),
          pairwise_ranking_accuracy=float(correct[comparable].float().mean()) if comparable.any() else None,
          comparable_pairs=int(comparable.sum()), best_second_margin=float(margin.mean()), tie_rate=float((margin<=tie_epsilon).float().mean()),
          expected_regret=float(((s/router_temperature).softmax(-1)*(y-y.amin(-1,keepdim=True))).sum(-1).mean()),
          uniform_random_regret=float((y.mean(-1)-y.amin(-1)).mean()),
          constant_per_step_regret=float((y[:,int(mean.argmin())]-y.amin(-1)).mean()),
          expert_utilization=torch.bincount(selected,minlength=y.size(1)).tolist(),
          unselected_experts=int((torch.bincount(selected,minlength=y.size(1))==0).sum()),
          value_prediction_error=None))  # Scores are preferences, not calibrated costs.
    result = dict(per_step=rows)
    for key in ('routing_regret','oracle_top1','oracle_top2','uniform_random_regret','constant_per_step_regret','expected_regret','tie_rate'):
        result[key] = sum(r[key] for r in rows)/len(rows)
    valid = [r for r in rows if r['comparable_pairs']]
    result['pairwise_ranking_accuracy'] = sum(r['pairwise_ranking_accuracy']*r['comparable_pairs'] for r in valid)/sum(r['comparable_pairs'] for r in valid) if valid else None
    return result


@torch.no_grad()
def descriptor_oracle_disagreement(descriptors, losses, *, near_threshold=0.05, top_k=2):
    """Held-out-to-held-out neighbours; inputs standardized with training stats."""
    n = descriptors.size(0)
    if n < 2:
        return dict(samples=n, available=False)
    distance = torch.cdist(descriptors.float(), descriptors.float()) / descriptors.size(-1)**.5
    distance.fill_diagonal_(float('inf'))
    nearest_distance, nearest = distance.min(-1)
    best = losses.argmin(-1)
    disagree = best != best[nearest]
    top = losses.topk(min(top_k, losses.size(-1)), largest=False).indices
    overlap = (top[:, :, None] == top[nearest, None, :]).any(-1).float().mean(-1)
    bins = []
    edges = [0., .01, .05, .1, .25, .5, 1., float('inf')]
    for low, high in zip(edges, edges[1:]):
        mask = (nearest_distance >= low) & (nearest_distance < high)
        bins.append(dict(min_distance=low, max_distance=high if high != float('inf') else None, count=int(mask.sum()),
                         oracle_disagreement=float(disagree[mask].float().mean()) if mask.any() else None))
    near = nearest_distance <= near_threshold
    return dict(samples=n, available=True, mean_nearest_distance=float(nearest_distance.mean()),
                nearest_distances=nearest_distance.tolist(), near_threshold=near_threshold, near_count=int(near.sum()),
                near_identical_disagreement=float(disagree[near].float().mean()) if near.any() else None,
                overall_nearest_disagreement=float(disagree.float().mean()), top_k_oracle_overlap=float(overlap.mean()),
                conditional_disagreement=bins)


def cache(router, bank):
    return [router.describe(s.latent) for s in bank.states]


def cached_scores(router, descriptors):
    return [router.score_descriptor(d, descriptors[t-1] if t else None)[0] for t,d in enumerate(descriptors)]


@torch.no_grad()
def profile_router(router, latent, repeats=10):
    rows=[]
    for _ in range(repeats):
        start=time.perf_counter()
        d=router.describe(latent, profile=True)
        if latent.is_cuda: torch.cuda.synchronize(latent.device)
        score_start=time.perf_counter();router.score_descriptor(d)
        if latent.is_cuda: torch.cuda.synchronize(latent.device)
        end=time.perf_counter()
        rows.append({**d.diagnostics['timing_seconds'], 'scoring':end-score_start,'total_routing':end-start})
    return {k:sum(r[k] for r in rows)/len(rows)*1000 for k in rows[0]}


def fit_comparison(train, held, config, *, updates=100, learning_rate=.01, seed=17, output=None,
                   progress_callback=None, cancellation_callback=None):
    torch.manual_seed(seed)
    c = spectral_config(config)
    means=[y.mean(0) for y in train.losses]
    variants=[('learned_geometric',None)]
    for size in (8,16,32):
        variants += [(f'{kind}_w{size}_b{basins}', replace(config,spectral_neighbourhood_size=size,
                       geometry_score_weight=alpha,resonance_score_weight=beta,basins_per_expert=basins))
                     for kind,alpha,beta,basins in [('spectral_only',0.,1.,4),('geometry_only',1.,0.,4),
                                                   ('combined',1.,1.,1),('combined',1.,1.,4)]]
    results=[]
    for name, variant in variants:
        torch.manual_seed(seed)
        if variant is None:
            router=GeometricNexusRouter(replace(config,router_type='geometric'))
            train_input=[s.latent.detach().mean(1,keepdim=True) for s in train.states]
            held_input=[s.latent.detach().mean(1,keepdim=True) for s in held.states]
            scoring=lambda inputs: [-router.route_one(x).base_actions[:,0] for x in inputs]
            objective=lambda scores: torch.stack([routing_objective(s,y,c) for s,y in zip(scores,train.losses)]).mean()
            config_used=replace(config,router_type="geometric")
        else:
            router=SpectralGeometricRouter(variant)
            train_input,held_input=cache(router,train),cache(router,held)
            # Initialization sees training descriptors only, with explicit deltas.
            g=torch.cat([router.geometry_vector(d,train_input[t-1] if t else None) for t,d in enumerate(train_input)])
            joined=Descriptor(torch.cat([d.q for d in train_input]), torch.cat([d.g for d in train_input]),{})
            router.initialize(joined,torch.cat(train.losses),geometry_vectors=g)
            scoring=lambda inputs: cached_scores(router,inputs)
            objective=lambda scores: torch.stack([routing_objective(s,y,router.c) for s,y in zip(scores,train.losses)]).mean()+router.c.basin_redundancy_weight*router.redundancy_loss()
            config_used=variant
        optimizer=torch.optim.AdamW(router.parameters(),lr=learning_rate,weight_decay=0)
        router.train()
        for update in range(updates):
            if cancellation_callback and cancellation_callback():
                from .training import TrainingCancelledError
                raise TrainingCancelledError('Spectral comparison cancelled between updates')
            optimizer.zero_grad(set_to_none=True)
            loss=objective(scoring(train_input))
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite bank fit')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            if progress_callback and (update == 0 or (update+1) % 10 == 0 or update+1 == updates):
                progress_callback(name, len(results)*updates+update+1, len(variants)*updates, results)
        router.eval()
        with torch.no_grad():
            report=dict(variant=name, parameters=sum(p.numel() for p in router.parameters()),
                        train=metrics(scoring(train_input),train.losses,means),
                        held_out=metrics(scoring(held_input),held.losses,means))
            if variant is not None:
                report['requested_window_size']=variant.spectral_neighbourhood_size
                report['actual_window_sizes']=[min(s.latent.size(1),variant.spectral_neighbourhood_size) for s in held.states]
                report['window_truncated']=any(s.latent.size(1)<variant.spectral_neighbourhood_size for s in held.states)
                report['latency_ms_batch1']=profile_router(router,held.states[0].latent[:1])
                occupancy=torch.zeros(config.num_modules,variant.basins_per_expert,dtype=torch.long)
                collisions=[]
                for t,d in enumerate(held_input):
                    previous=held_input[t-1] if t else None
                    scores,details=router.score_descriptor(d,previous)
                    selected=scores.argmax(-1)
                    basin=details['winning_basin'].gather(1,selected[:,None]).squeeze(1)
                    occupancy.view(-1).add_(torch.bincount(selected*variant.basins_per_expert+basin,minlength=occupancy.numel()))
                    full=torch.cat((router.standardizer(router.geometry_vector(d,previous)),d.q),-1)
                    collisions.append(descriptor_oracle_disagreement(full,held.losses[t]))
                report.update(basin_occupancy=occupancy.tolist(),dead_basins=int((occupancy==0).sum()),
                              basin_redundancy=float(router.redundancy_loss()),descriptor_oracle_disagreement=collisions)
            else:
                start=time.perf_counter()
                for _ in range(20): router.route_one(held_input[0][:1])
                report['latency_ms_batch1']={'total_routing':(time.perf_counter()-start)/20*1000}
        if output:
            torch.save(dict(router=router.state_dict(),config=asdict(config_used),variant=name),Path(output)/(name+'.pt'))
        results.append(report)
        if progress_callback:
            progress_callback(name, len(results)*updates, len(variants)*updates, results)
    return results


@torch.no_grad()
def transformation_audit(reference, train, held, config):
    c=spectral_config(config)
    standardizer=RunningStandardizer(len(transformation_names(c)),c.geometry_std_floor)
    def measure(bank):
        rows=[]
        for state in bank.states:
            signatures=[]
            for expert in range(config.num_modules):
                selected=torch.full((state.latent.size(0),),expert,dtype=torch.long)
                after=reference.apply_expert(state.latent,selected)
                signatures.append(transformation_signature(state.latent,after,c).signature)
            rows.append(torch.stack(signatures,1))
        return rows
    training=measure(train)
    standardizer.update(torch.cat(training));standardizer.frozen.fill_(True);standardizer.eval()
    rows=[]
    for signature,loss in zip(measure(held),held.losses):
        value,similarity,overlap=complementarity(signature,(-loss/c.spectral_target_temperature).softmax(-1),
                                                c.transformation_complementarity_threshold,standardizer=standardizer)
        rows.append(dict(observational_loss=float(value),mean_pairwise_similarity=similarity.mean(0).tolist(),
                         mean_usefulness_overlap=overlap.mean(0).tolist(),mean_signature=signature.mean(0).tolist()))
    return dict(mode='observational',feature_names=transformation_names(c),per_step=rows,
                normalization='training-only population statistics')


def smoke_banks(seed=17, count=24):
    torch.manual_seed(seed)
    config=EMCConfig(latent_dim=8,num_modules=3,modules_per_cycle=1,num_cycles=2,trajectory_steps=2,
        vocab_size=16,max_sequence_length=32,module_hidden_dim=16,attention_heads=2,integrator_heads=2,
        module_families=('gpt','recurrent','delta'),architecture_stage='n1_sequential',
        router_type='counterfactual_value',integrator_type='identity_free_gate',refractory_enabled=False,
        loss_free_balance_enabled=False,switch_cost=0.,persistence_bonus=0.,routing_geometry_dim=8)
    reference=CounterfactualValueEMC(config)
    # Brief uniform expert development on a synthetic copy-next-token objective.
    optimizer=torch.optim.Adam(reference.parameters(),lr=.003)
    generator=torch.Generator().manual_seed(seed+1)
    for step in range(30):
        x=torch.randint(0,16,(4,32),generator=generator)
        state=reference.embed(x)
        selected=torch.full((4,),step%3,dtype=torch.long)
        logits=reference.read_endpoint(reference.apply_expert(state,selected))
        loss=F.cross_entropy(logits,x[:,-1])
        optimizer.zero_grad();loss.backward();optimizer.step()
    reference.eval()
    x=torch.randint(0,16,(count,32),generator=generator); vx=torch.randint(0,16,(count,32),generator=generator)
    train=measure_bank(reference,x,x,generator)
    held=measure_bank(reference,vx,vx,generator)
    return train,held,replace(config,router_type='spectral_geometric'),reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--bank',type=Path);group.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=100)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--prefixes',type=int,default=24)
    parser.add_argument('--geometry-dim',type=int,default=8,help='learned geometric baseline need dimension')
    parser.add_argument('--reference-checkpoint',type=Path,help='optional bank source checkpoint for transformation audit')
    args=parser.parse_args()
    if args.geometry_dim<=0: parser.error('geometry dimension must be positive')
    if args.updates<=0 or args.prefixes<2: parser.error('positive updates and at least two prefixes required')
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
    reference=None
    if args.smoke:
        train,held,config,reference=smoke_banks(args.seed,args.prefixes)
        payload=dict(train=train.payload(),held_out=held.payload(),source='synthetic_copy_smoke')
        torch.save(payload,args.output/'smoke-bank.pt')
        source='synthetic_copy_smoke; NOT real-data validation'
    else:
        payload=torch.load(args.bank,map_location='cpu',weights_only=True)
        if 'source_identity' not in payload: raise ValueError('bank lacks fixed-reference source identity')
        data=payload['train'];n,length,dim=data['states'][0].shape;e=data['losses'][0].size(1)
        config=EMCConfig(latent_dim=dim,num_modules=e,modules_per_cycle=1,num_cycles=len(data['states']),
          trajectory_steps=len(data['states']),attention_heads=1,integrator_heads=1,delta_heads=1,
          architecture_stage='n1_sequential',router_type='spectral_geometric',max_sequence_length=length,
          loss_free_balance_enabled=False)
        class ReferenceShape:
            pass
        shape=ReferenceShape();shape.config=config
        config=replace(config,routing_geometry_dim=args.geometry_dim)
        shape.config=config
        train=load_bank_payload(data,shape,n,length,'cpu')
        held=load_bank_payload(payload['held_out'],shape,n,length,'cpu')
        source=payload['source_identity']
        if args.reference_checkpoint:
            from .checkpoint import load_model_checkpoint
            from .value_fit import source_identity
            loaded=load_model_checkpoint(args.reference_checkpoint)
            reference=loaded.model.eval().float()
            if not isinstance(reference,CounterfactualValueEMC):
                raise ValueError('source audit currently requires the value FitBank reference model')
            if 'reference_router' in payload:
                reference.router.load_state_dict(payload['reference_router'])
            if source_identity(reference,loaded.tokenizer,length)!=source:
                raise ValueError('reference checkpoint does not match saved bank source identity')
    if set(map(tuple,train.prefixes.tolist())) & set(map(tuple,held.prefixes.tolist())):
        raise ValueError('training/held-out prefixes overlap')
    digest=hashlib.sha256()
    for bank in (train,held):
        for x in (bank.prefixes,*(s.latent for s in bank.states),*bank.losses): digest.update(x.contiguous().numpy().tobytes())
    config=replace(config,routing_geometry_dim=args.geometry_dim)
    results=fit_comparison(train,held,config,updates=args.updates,seed=args.seed,output=args.output)
    report=dict(source=source,bank_sha256=digest.hexdigest(),seed=args.seed,updates=args.updates,
                config=asdict(config),results=results,conclusion='INCONCLUSIVE',
                caveat='Small controlled fit; no claim of reliable specialization. Scores are not cost predictions.',
                transformation=transformation_audit(reference,train,held,config) if reference else
                  {'available':False,'reason':'existing FitBank stores states/losses, not transformed outputs; rerun audit with source reference'})
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{ 'variant':r['variant'], 'held_regret':r['held_out']['routing_regret'], 'top1':r['held_out']['oracle_top1']} for r in results],indent=2))


if __name__=='__main__':main()
