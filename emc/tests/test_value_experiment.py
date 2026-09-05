from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch

from rayvan_emc.checkpoint import load_model_checkpoint
from rayvan_emc.data import LanguageCorpus
from rayvan_emc.research_config import ExperimentConfig, ModelConfig, ResearchTrainingConfig, RoutingConfig, research_schema
from rayvan_emc.research_runner import _build_model, estimate_experiment, run_experiment
from rayvan_emc.training import TrainingConfig, train_model, evaluate_model_metrics
from rayvan_emc.value_routing import CounterfactualValueEMC


def experiment(**updates):
    c = ExperimentConfig(architecture='counterfactual_value_emc',suite='capability_10',experts={'delta':2},
                         routing=RoutingConfig(routing_geometry_dim=8, value_probe_rate=1.,value_development_batch_size=2),
                         model=ModelConfig(preset='custom',latent_dim=8,context_length=8,attention_heads=2,
                                           integrator_heads=2,module_hidden_dim=16),
                         training=ResearchTrainingConfig(tokens=4,batch_size=2,evaluation_interval=1,evaluation_batches=1,
                                                         device='cpu',precision='fp32',diagnostic_examples_per_capability=1))
    return replace(c, **updates)


def test_config_roundtrip_keeps_new_algorithm_explicit_and_legacy_unchanged():
    c = experiment()
    normalized = c.to_dict()
    assert ExperimentConfig.from_dict(normalized).to_dict() == normalized
    assert normalized['routing']['value_target'] == 'suffix'
    assert 'top_k' not in normalized['routing']
    assert 'refractory_enabled' not in normalized['routing']
    assert 'value_target' not in ExperimentConfig().to_dict()['routing']
    assert estimate_experiment(c)['architecture'] == 'counterfactual_value_emc'
    assert any(row['id']=='counterfactual_value_emc' for row in research_schema()['architectures'])


@pytest.mark.parametrize('settings', [dict(value_common_fraction=-.1),dict(value_probe_rate=1.1),
                                     dict(value_specialist_temperature=0),dict(value_development_interval=0),
                                     dict(value_target='unknown'),dict(value_expert_training='balanced')])
def test_invalid_value_settings_fail_before_launch(settings):
    with pytest.raises(ValueError):
        experiment(routing=RoutingConfig(**settings))


def test_checkpoint_resume_restores_expert_clocks_and_sampling_exactly(tmp_path):
    torch.set_num_threads(1)
    corpus = LanguageCorpus.from_texts(['ababacabbc\n'*10], ['bacabbcaba\n'*10])
    c = experiment()
    initial = _build_model(c, corpus.tokenizer.vocab_size)
    full, part = deepcopy(initial), deepcopy(initial)
    train = TrainingConfig(steps=2,batch_size=2,sequence_length=4,learning_rate=.001,
                           evaluation_interval=1,evaluation_batches=1,seed=42,router_balance_coefficient=0.,
                           checkpoint_directory=str(tmp_path/'full'), retain_milestone_checkpoints=False)
    full_result = train_model(full,corpus,train,print_progress=False)
    first = train_model(part,corpus,replace(train,steps=1,checkpoint_directory=str(tmp_path/'part')),print_progress=False)
    loaded = load_model_checkpoint(first.latest_checkpoint)
    assert isinstance(loaded.model,CounterfactualValueEMC)
    resumed = train_model(loaded.model,corpus,replace(train,resume_from=first.latest_checkpoint,
                         checkpoint_directory=str(tmp_path/'resumed')),print_progress=False)
    for name, p in full.state_dict().items():
        torch.testing.assert_close(p, loaded.model.state_dict()[name],rtol=0,atol=0)
    assert resumed.module_diagnostics['value_routing']['expert_update_batches']==[2,2]
    assert resumed.tokens_processed==full_result.tokens_processed==4
    assert resumed.module_diagnostics['value_routing']['costs']==full_result.module_diagnostics['value_routing']['costs']
    assert evaluate_model_metrics(loaded.model,corpus,train)[0] > 0


def test_backend_saves_endpoint_budget_opportunity_and_heldout_metrics(tmp_path):
    torch.set_num_threads(1)
    summary = run_experiment(experiment(), runs_directory=tmp_path,run_id='value-e2e')
    assert summary['status']=='completed'
    assert summary['model']['objective']=='prefix_endpoint'
    assert summary['training_result']['tokens_processed']==4
    value = summary['value_routing']
    assert value['expert_update_batches']==[2,2]
    assert value['expert_training_items']==[4,4]
    assert value['costs']['input_tokens']==32
    assert value['held_out']['target']=='held_out_suffix'
    assert len(value['held_out']['by_depth'])==3
    events = [json.loads(line) for line in (tmp_path/'value-e2e'/'metrics.jsonl').read_text().splitlines()]
    training = [row for row in events if row['type']=='training_step']
    assert len(training)==2  # No duplicate event when validation coincides.
    assert all(row['objective']=='prefix_endpoint' for row in training)
    assert all('held_out' in row['routing']['value_routing'] for row in training)
    assert not any(w.get('code')=='collapsed_routing' for row in events for w in row.get('warnings',[]))
