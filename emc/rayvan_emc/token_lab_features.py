"""Observational measurements. No trainable geometry or routing dependencies."""
import math
import time
import torch
from torch.nn import functional as F

EPS = 1e-8
SCHEMA = {}


def register(key, category, definition, units='dimensionless', role='input', availability='finite nondegenerate inputs', expensive=False):
    SCHEMA[key] = dict(key=key, display_name=key.replace('_',' '), category=category,
        definition=definition, interpretation=definition, units=units, role=role,
        availability=availability, expensive=expensive)


def scalar(x):
    if x is None: return None
    v = float(x)
    return v if math.isfinite(v) else None


def cosine(a,b):
    if a.norm() <= EPS or b.norm() <= EPS: return None
    return scalar(F.cosine_similarity(a.reshape(1,-1),b.reshape(1,-1))[0])


def stats(x):
    return dict(mean=scalar(x.mean()), std=scalar(x.std(unbiased=False)),
                norm=scalar(x.norm()), rms=scalar(x.square().mean().sqrt()),
                max_abs=scalar(x.abs().max()), near_zero=scalar((x.abs()<1e-5).float().mean()))


for k,definition in dict(mean='mean(h)',std='population std(h)',norm='L2 norm(h)',rms='sqrt(mean(h^2))',max_abs='max(abs(h))',near_zero='fraction abs(h)<1e-5').items():
    register('hidden_'+k,'Hidden state',definition,'latent units' if k!='near_zero' else 'fraction')
for k,definition in {
    'skewness':'mean((h-mean(h))^3)/std(h)^3', 'kurtosis':'mean((h-mean(h))^4)/std(h)^4 - 3'
}.items(): register('hidden_'+k,'Hidden state',definition,availability='std(h)>1e-8')

DEFINITIONS = {
 'previous_distance':('Dynamics','norm(h_t-h_(t-1))','latent units'),
 'velocity':('Dynamics','same as previous_distance; representation change, not physical velocity','latent units'),
 'previous_cosine':('Dynamics','cos(h_t,h_(t-1))','cosine'),
 'acceleration':('Dynamics','norm(h_t-2h_(t-1)+h_(t-2))','latent units'),
 'displacement_mean':('Dynamics','mean consecutive L2 distances within causal window','latent units'),
 'displacement_std':('Dynamics','population std consecutive L2 distances','latent units'),
 'consecutive_cosine':('Dynamics','mean cosine of consecutive nonzero states','cosine'),
 'centroid_distance':('Dispersion','norm(h_t-mean(H))','latent units'),
 'radius':('Dispersion','sqrt(mean_j norm(h_j-mean(H))^2)','latent units'),
 'pair_distance_mean':('Dispersion','mean distances over unordered distinct pairs','latent units'),
 'pair_distance_std':('Dispersion','population std unordered pair distances','latent units'),
 'pair_distance_max':('Dispersion','maximum unordered pair distance','latent units'),
 'pair_distance_min_positive':('Dispersion','minimum pair distance >1e-8','latent units'),
 'pair_cosine_mean':('Dispersion','mean unordered pair cosine, nonzero vectors only','cosine'),
 'pair_cosine_std':('Dispersion','population std unordered pair cosine','cosine'),
 'current_cosine_mean':('Dispersion','mean cosine(current, preceding window states)','cosine'),
 'current_cosine_min':('Dispersion','min cosine(current, preceding window states)','cosine'),
 'current_cosine_max':('Dispersion','max cosine(current, preceding window states)','cosine'),
 'effective_rank':('Shape','1/sum(p_k^2), p_k=sigma_k^2/sum(sigma^2); dimension proxy','dimensions'),
 'pca_entropy':('Shape','-sum(p_k log(p_k))','nats'),
 'pca_entropy_normalized':('Shape','pca_entropy/log(min(m-1,D))','fraction'),
 'anisotropy':('Shape','sigma_1/mean(sigma), retaining min(m-1,D) singular values','ratio'),
 'variance':('Shape','sum(sigma^2)/m; population covariance trace','latent units squared'),
 'condition':('Shape','sigma_1/sigma_r; null if rank deficient relative to min(m-1,D)','ratio'),
 'nonflatness':('Shape','sum energy after first min(2,r) PCA components / total energy; not curvature','fraction'),
 'nearest_distance':('Density','minimum distance from current to another window state','latent units'),
 'knn_mean':('Density','mean of k nearest distances from current to preceding window states','latent units'),
 'knn_median':('Density','median of those k distances','latent units'),
 'knn_std':('Density','population std of those k distances','latent units'),
 'density':('Density','1/(knn_mean+1e-8)','inverse latent units'),
}
for k,(cat,d,u) in DEFINITIONS.items(): register(k,cat,d,u)
for i in range(4): register(f'pca_p{i+1}','Shape',f'sigma_{i+1}^2/sum(sigma^2); null if beyond min(m-1,D)','fraction')


def core(h,k=4):
    h=h.detach().float().cpu(); m,d=h.shape; x=h-h.mean(0); last=h[-1]
    out={key:None for key in SCHEMA if SCHEMA[key]['category'] in ['Hidden state','Dynamics','Dispersion','Shape','Density']}
    out.update({'hidden_'+a:b for a,b in stats(last).items()})
    std=last.std(unbiased=False)
    if std>EPS:
        z=(last-last.mean())/std;out.update(hidden_skewness=scalar(z.pow(3).mean()),hidden_kurtosis=scalar(z.pow(4).mean()-3))
    out.update(centroid_distance=scalar(x[-1].norm()),radius=scalar(x.square().sum(-1).mean().sqrt()))
    if m>1:
        dx=(h[1:]-h[:-1]).norm(dim=-1);pd=torch.pdist(h)
        out.update(previous_distance=scalar(dx[-1]),velocity=scalar(dx[-1]),previous_cosine=cosine(h[-1],h[-2]),
            displacement_mean=scalar(dx.mean()),displacement_std=scalar(dx.std(unbiased=False)),
            pair_distance_mean=scalar(pd.mean()),pair_distance_std=scalar(pd.std(unbiased=False)),pair_distance_max=scalar(pd.max()),
            pair_distance_min_positive=scalar(pd[pd>EPS].min()) if (pd>EPS).any() else None)
        cs=[cosine(a,b) for a,b in zip(h[1:],h[:-1])];cs=[v for v in cs if v is not None]
        out['consecutive_cosine']=sum(cs)/len(cs) if cs else None
        n=F.normalize(h,dim=-1);mat=n@n.T; mask=torch.triu(torch.ones(m,m,dtype=torch.bool),1)&(h.norm(dim=-1)>EPS)[:,None]&(h.norm(dim=-1)>EPS)[None,:]
        if mask.any():out.update(pair_cosine_mean=scalar(mat[mask].mean()),pair_cosine_std=scalar(mat[mask].std(unbiased=False)))
        cs=[cosine(last,v) for v in h[:-1]];cs=[v for v in cs if v is not None]
        if cs:out.update(current_cosine_mean=sum(cs)/len(cs),current_cosine_min=min(cs),current_cosine_max=max(cs))
        kn=(h[:-1]-last).norm(dim=-1).topk(min(k,m-1),largest=False).values
        out.update(nearest_distance=scalar(kn.min()),knn_mean=scalar(kn.mean()),knn_median=scalar(kn.median()),knn_std=scalar(kn.std(unbiased=False)),density=scalar(1/(kn.mean()+EPS)))
        s=torch.linalg.svdvals(x)[:min(m-1,d)];energy=s.square().sum();out['variance']=scalar(energy/m)
        if energy>EPS:
            p=s.square()/energy;ent=-(p*p.clamp_min(EPS).log()).sum()
            out.update(effective_rank=scalar(1/p.square().sum()),pca_entropy=scalar(ent),
                pca_entropy_normalized=scalar(ent/math.log(len(s))) if len(s)>1 else None,
                anisotropy=scalar(s[0]/s.mean()),condition=scalar(s[0]/s[-1]) if s[-1]>s[0]*1e-6 else None,
                nonflatness=scalar(p[2:].sum()))
            out.update({f'pca_p{i+1}':scalar(p[i]) if i<len(p) else None for i in range(4)})
    if m>2:out['acceleration']=scalar((h[-1]-2*h[-2]+h[-3]).norm())
    return out


for k,d in {'baseline_loss':'-log p(target)','predictive_entropy':'-sum_v p(v) log p(v)', 'top1_probability':'max p(v)',
 'target_probability':'p(target)', 'logit_margin':'largest minus second largest logit', 'target_margin':'target logit minus largest logit (zero when target is top1)'}.items():
    register(k,'Prediction',d,'nats' if k in ['baseline_loss','predictive_entropy','logit_margin','target_margin'] else 'probability',role='outcome' if k in ['baseline_loss','target_probability','target_margin'] else 'input')


def prediction(logits,target):
    logits=logits.detach().float().cpu();lp=logits.log_softmax(-1);p=lp.exp();top=logits.topk(2).values
    return dict(baseline_loss=scalar(-lp[target]),predictive_entropy=scalar(-(p*lp).sum()),top1_probability=scalar(p.max()),
                target_probability=scalar(p[target]),logit_margin=scalar(top[0]-top[1]),target_margin=scalar(logits[target]-top[0]))


NOVEL_DESCRIPTOR=['radius','effective_rank','pca_entropy','nonflatness','pair_distance_mean','knn_mean','hidden_norm','centroid_distance']
for k,d in {'hidden_novelty_nearest':'nearest L2 distance of L2-normalized h to training reference states',
 'hidden_novelty_knn':'mean k-nearest normalized hidden distance', 'reference_max_cosine':'maximum cosine to training reference states',
 'cosine_novelty':'1-reference_max_cosine', 'descriptor_novelty_nearest':'nearest training-standardized descriptor L2 distance',
 'descriptor_novelty_knn':'mean k nearest training-standardized descriptor L2 distance',
 'success_familiarity':'nearest standardized descriptor distance to training-reference cases with this expert improvement > tie_epsilon'}.items():register(k,'Novelty',d,availability='nonempty training-only reference bank; complete descriptors',role='input')


def novelty(h,features,bank,k=4):
    out={key:None for key in SCHEMA if SCHEMA[key]['category']=='Novelty'}
    if h.norm()>EPS and len(bank['hidden']):
        z=F.normalize(h.detach().float().cpu(),dim=0);ref=bank['hidden'];dist=(ref-z).norm(dim=-1)
        out.update(hidden_novelty_nearest=scalar(dist.min()),hidden_novelty_knn=scalar(dist.topk(min(k,len(dist)),largest=False).values.mean()),reference_max_cosine=scalar((ref@z).max()),cosine_novelty=scalar(1-(ref@z).max()))
    if all(features.get(key) is not None for key in NOVEL_DESCRIPTOR) and len(bank['descriptor']):
        z=(torch.tensor([features[key] for key in NOVEL_DESCRIPTOR])-bank['mean'])/bank['std']
        dist=(bank['descriptor']-z).norm(dim=-1)
        out.update(descriptor_novelty_nearest=scalar(dist.min()),descriptor_novelty_knn=scalar(dist.topk(min(k,len(dist)),largest=False).values.mean()))
        if len(bank.get('success',[])):out['success_familiarity']=scalar((bank['success']-z).norm(dim=-1).min())
    return out


for k,d in {'delta_norm':'norm(delta)', 'relative_delta':'norm(delta)/(norm(h_before)+1e-8)', 'input_output_distance':'norm(h_after-h_before)',
 'input_delta_cosine':'cos(h_before,delta)', 'input_output_cosine':'cos(h_before,h_after)', 'norm_change':'norm(h_after)-norm(h_before)'}.items():register(k,'Transformation',d,role='response')
for k in list(SCHEMA):
    if SCHEMA[k]['category'] in ['Hidden state','Dynamics','Dispersion','Shape','Density','Prediction']:
        register('change_'+k,'Transformation','after minus before: '+SCHEMA[k]['definition'],SCHEMA[k]['units'],role='response')

for k,d in {'gradient_norm':'norm(d baseline endpoint NLL / d current latent)', 'gradient_abs_mean':'mean(abs(gradient))',
 'gradient_abs_max':'max(abs(gradient))','gradient_delta_cosine':'cos(gradient,delta)', 'negative_gradient_delta_cosine':'cos(-gradient,delta)'}.items():register(k,'Gradient',d,role='outcome',expensive=True,availability='deep tier selected and Bernoulli sampled')

for k,d in {'entropy_mean':'mean head -sum a log(a)', 'entropy_std':'std head entropy','entropy_min':'min head entropy','entropy_max':'max head entropy',
 'entropy_normalized':'mean entropy/log(prefix length)', 'effective_tokens':'mean head exp(entropy)', 'max_weight':'mean head max(a)',
 'concentration':'mean head sum(a^2)', 'distance':'mean head sum_j a_j*(t-j)'}.items():register('attention_'+k,'Attention',d,role='response',availability='GPT-style expert attention only')
for phase in ['pre','post']:
    for k,d in dict(mean='mean(a)',std='population std(a)',norm='norm(a)',near_zero='fraction abs(a)<1e-5',strong='fraction abs(a)>1',max='max(a)',kurtosis='mean standardized(a)^4 - 3',participation='sum(a^2)^2/sum(a^4)').items():register(f'ffn_{phase}_{k}','FFN',d,role='response',availability='conventional expert GELU FFN hook only')


def instrumentation(module,state):
    """Sampled only; hook aggregates last-token FFN, discard arrays immediately."""
    out={k:None for k,v in SCHEMA.items() if v['category'] in ['Attention','FFN']};handles=[]
    def hook(_m,inputs,output):
        for name,a in [('pre',inputs[0][0,-1].detach().float()),('post',output[0,-1].detach().float())]:
            s=stats(a);out.update({f'ffn_{name}_{k}':s[k] for k in ['mean','std','norm','near_zero']})
            out.update({f'ffn_{name}_strong':scalar((a.abs()>1).float().mean()),f'ffn_{name}_max':scalar(a.max()),
              f'ffn_{name}_participation':scalar(a.square().sum().square()/a.pow(4).sum()) if a.pow(4).sum()>EPS else None,
              f'ffn_{name}_kurtosis':scalar(((a-a.mean())/a.std(unbiased=False)).pow(4).mean()-3) if a.std(unbiased=False)>EPS else None})
    for m in module.modules():
        if isinstance(m,torch.nn.GELU):handles.append(m.register_forward_hook(hook))
    try:
        with torch.no_grad():
            if state.is_cuda:torch.cuda.synchronize(state.device)
            started=time.perf_counter()
            delta=module(state)
            if state.is_cuda:torch.cuda.synchronize(state.device)
            out['_instrumented_expert_seconds']=time.perf_counter()-started
            if getattr(module,'family',None)=='gpt':
                n=module.attention_norm(state);t=n.size(1)
                _,a=module.attention(n,n,n,attn_mask=torch.ones(t,t,dtype=torch.bool,device=n.device).triu(1),need_weights=True,average_attn_weights=False)
                a=a[0,:,-1].float();ent=-(a*a.clamp_min(EPS).log()).sum(-1)
                vals=dict(entropy_mean=ent.mean(),entropy_std=ent.std(unbiased=False),entropy_min=ent.min(),entropy_max=ent.max(),
                    entropy_normalized=ent.mean()/math.log(t) if t>1 else None,effective_tokens=ent.exp().mean(),max_weight=a.max(-1).values.mean(),concentration=a.square().sum(-1).mean(),distance=(a*torch.arange(t-1,-1,-1,device=a.device)).sum(-1).mean())
                out.update({'attention_'+k:scalar(v) for k,v in vals.items()})
    finally:
        for handle in handles:handle.remove()
    return delta,out


def outcomes(before,losses,independent,eps):
    vals=torch.tensor(losses);order=vals.argsort();best=float(vals.min());margin=float(vals[order[1]]-vals[order[0]]) if len(vals)>1 else None
    return [dict(expert_loss=float(v),improvement=float(before[i]-v),relative_advantage=float((vals.sum()-v)/(len(vals)-1)-v) if independent and len(vals)>1 else None,
        oracle_best_expert=int(order[0]) if independent else None,expert_rank=int((vals<v-eps).sum())+1 if independent else None,
        best_second_margin=margin if independent else None,tied=bool((vals<=best+eps).sum()>1) if independent and len(vals)>1 else None,
        is_best=bool(v<=best+eps) if independent else None) for i,v in enumerate(vals)]

for k in ['expert_loss','improvement','relative_advantage','expert_rank','best_second_margin']:
    register(k,'Outcome',{'expert_loss':'post expert endpoint NLL','improvement':'baseline_loss-expert_loss','relative_advantage':'mean other expert losses minus this loss, same input only','expert_rank':'1 + count(loss_j < loss_i-tie_epsilon)','best_second_margin':'second smallest minus smallest same-input loss'}[k],role='outcome')


def spectral(h):
    from .spectral_config import SpectralConfig
    from .spectral_geometry import extract_descriptor, feature_names, centre_normalize, build_graph, decompose_graph
    c=SpectralConfig(spectral_neighbourhood_size=len(h));h=h.detach().float().cpu()[None]
    d=extract_descriptor(h,c);names,_=feature_names(c)
    out={'spectral_'+name:scalar(v) for name,v in zip(names,d.q[0])}
    x,_,_=centre_normalize(h,EPS);a,_,_,_=build_graph(x,c);ev,_,_=decompose_graph(a,c);ev=ev[0];positive=ev[ev>c.spectral_zero_threshold]
    out.update({f'laplacian_lambda_{i+1}':scalar(positive[i]) if i<len(positive) else None for i in range(4)})
    out['laplacian_gap']=scalar(ev[1]) if len(ev)>1 else None
    out['laplacian_zero_modes']=int((ev<=c.spectral_zero_threshold).sum())
    p=ev/ev.sum().clamp_min(EPS);out['laplacian_entropy']=scalar(-(p*p.clamp_min(EPS).log()).sum()) if ev.sum()>EPS else None
    return out


def initialize_spectral_schema():
    from .spectral_config import SpectralConfig
    from .spectral_geometry import feature_names
    names,_=feature_names(SpectralConfig())
    for name in names:
        if name.startswith('spectrum.'):
            definition='sum_retained exp(-0.5*((log(lambda)-band_center)/0.8)^2), normalized by total band mass'
        elif '.band_' in name:
            definition='sum_retained ((u_k^T centered_signal)^2 * exp(-0.5*((log(lambda_k)-band_center)/0.8)^2)) / total retained signal energy; signal='+name.split('.')[0]
        elif name.startswith('hks.'):
            definition='vertex '+name.split('.')[1]+' of sum_k exp(-t*lambda_k)*u_k(vertex)^2; retained nontrivial and zero modes; 6 log times 0.1..100'
        else:
            definition='vertex '+name.split('.')[1]+' of sum_retained normalized_log_frequency_filter(lambda_k)*u_k(vertex)^2; 6 log bands 0.01..2'
        register('spectral_'+name,'Spectral',definition+'; then divide by L2 norm of concatenated 48-component signature. Graph uses centered/RMS-normalized cloud, symmetric kNN k=4 with whole distance ties, Gaussian bandwidth median positive selected distance; retain first 8 nontrivial modes including boundary eigenspace.',expensive=True,availability='spectral tier sampled')
    for i in range(4):register(f'laplacian_lambda_{i+1}','Spectral',f'{i+1}th eigenvalue >1e-5 of symmetric normalized kNN Laplacian',expensive=True)
    register('laplacian_gap','Spectral','second smallest normalized Laplacian eigenvalue, including zero modes',expensive=True)
    register('laplacian_zero_modes','Spectral','count eigenvalues <=1e-5; component proxy',expensive=True)
    register('laplacian_entropy','Spectral','-sum p log p, p=lambda/sum(lambda), all modes',units='nats',expensive=True)


initialize_spectral_schema()

for key,d in {'position':'zero-based position within observed prefix','source_position':'zero-based token position in source block/example','context_start':'source block/example offset of observed prefix','window_size':'actual min(configured window, observed prefix length)',
 'position_normalized':'position / configured maximum sequence length','token_count':'count in training/reference corpus','token_frequency':'(training count+1)/(total training count+vocab size)',
 'log_token_frequency':'log smoothed training token frequency','rarity':'-log smoothed training token frequency','token_char_length':'length of decoded standalone token; tokenizer-dependent control'}.items():register(key,'Token',d)
for key in ['word_begin','word_end']:register(key,'Token','null: tokenizer does not expose reliable standalone word boundaries',availability='unavailable')
for key in list(SCHEMA):
    if SCHEMA[key]['category']=='Spectral':register('change_'+key,'Transformation','after minus before '+SCHEMA[key]['definition'],role='response',expensive=True)
register('change_attention_entropy','Transformation','unavailable: there is no common attention operator shared by before/after states of heterogeneous stages',role='response',availability='not defined for this protocol')
for key in ['delta_norm','input_output_distance','norm_change']:SCHEMA[key]['units']='latent units'
for key in ['gradient_norm','gradient_abs_mean','gradient_abs_max']:SCHEMA[key]['units']='nats per latent unit'
for key in ['attention_entropy_mean','attention_entropy_std','attention_entropy_min','attention_entropy_max']:SCHEMA[key]['units']='nats'
for key in ['attention_effective_tokens','attention_distance','position','source_position','context_start','window_size']:SCHEMA[key]['units']='tokens'
SCHEMA['token_count']['units']='occurrences'
SCHEMA['token_char_length']['units']='decoded characters'
for key in ['log_token_frequency','rarity']:SCHEMA[key]['units']='nats'

register('relative_improvement','Outcome','improvement/(abs(baseline_loss)+1e-8)',role='outcome')
register('successful','Outcome','1 if improvement > 0 else 0',role='outcome')
# Origin and ground-truth access are orthogonal. Preserve legacy role for old runs.
for key,item in SCHEMA.items():
    item['requires_target']=key in ['baseline_loss','target_probability','target_margin'] or item['role']=='outcome' or key.startswith('change_target')
    item['origin']=('expert_history' if key=='success_familiarity' else
        'expert_internal' if item['category'] in ['Attention','FFN','Gradient'] else
        'pre_expert' if item['role']=='input' or key in ['baseline_loss','target_probability','target_margin'] else 'post_expert')


def blind_keys(require_inference_available=False):
    """Allowlist enforced in backend; unknown keys and task metadata never qualify."""
    return [k for k,v in SCHEMA.items() if v['origin']=='pre_expert' and (not require_inference_available or not v['requires_target'])]
