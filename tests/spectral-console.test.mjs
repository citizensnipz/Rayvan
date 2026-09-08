import {test,after} from 'node:test';
import assert from 'node:assert/strict';
import {JSDOM} from 'jsdom';
import React,{act,useState} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
const dom=new JSDOM('<!doctype html><div id="root"></div>',{url:'http://localhost'});
Object.assign(globalThis,{window:dom.window,document:dom.window.document,HTMLElement:dom.window.HTMLElement,IS_REACT_ACT_ENVIRONMENT:true});
const {createRoot}=await import('react-dom/client');
const {ExperimentBuilder}=await import('../src/research/ExperimentBuilder.tsx');
const {SpectralLivePanel}=await import('../src/research/SpectralLivePanel.tsx');
const root=createRoot(document.getElementById('root'));
after(async()=>{await act(()=>root.unmount());dom.window.close();});

test('UI activates sequential test, preserves local paths and submits actual settings',async()=>{
 const initial={schema_version:3,name:'heterogeneous',notes:'',tags:[],projection_targets:[],suite:'capability_10',architecture:'counterfactual_value_emc',
  experts:{gpt:2,ssm:2,recurrent:2,delta:2},model:{preset:'quick',latent_dim:64,module_hidden_dim:128,context_length:256,attention_heads:4,integrator_heads:4,ssm_backend:'auto'},
  routing:{value_checkpoint_path:'C:/local/model-latest.pt',value_fit_bank_path:'C:/local/router-fit-bank.pt',value_spectral_comparison:true,
   value_expert_training:'frozen',value_fit_enabled:true,value_head_type:'linear',value_fit_prefixes:256,value_fit_updates:100,trajectory_steps:3,routing_geometry_dim:8,value_router_seed:17},
  training:{tokens:1000000,batch_size:4,learning_rate:.01,weight_decay:0,seed:42,gradient_accumulation:1,precision:'fp32',device:'cuda',evaluation_interval:20}};
 const schema={defaults:initial,suites:[{id:'capability_10',label:'10 tasks',tasks:[]}],architectures:[{id:'counterfactual_value_emc',label:'Value EMC'}],presets:{},model_presets:{},expert_families:Object.keys(initial.experts).map(id=>({id,label:id}))};
 let current,submitted;
 function App(){const [config,setConfig]=useState(initial);current=config;return React.createElement(ExperimentBuilder,{schema,config,setConfig,estimating:false,active:false,onRun:()=>{submitted=config;}});}
 await act(()=>root.render(React.createElement(App)));
 const toggle=[...document.querySelectorAll('label')].find(n=>n.textContent.includes('sequential learning + live trajectory charts')).querySelector('input');
 await act(()=>toggle.click());
 assert.equal(current.routing.value_spectral_live,true);
 assert.equal(current.routing.value_spectral_comparison,false);
 assert.equal(current.routing.value_checkpoint_path,initial.routing.value_checkpoint_path);
 assert.equal(current.routing.value_fit_bank_path,initial.routing.value_fit_bank_path);
 assert.deepEqual(current.model,initial.model);
 assert.deepEqual(current.experts,initial.experts);
 assert.equal(current.training.seed,42);
 for(const label of ['Neighbourhood window (tokens)','Basins per expert','Bank warmup updates','Sequential learning updates','Sequential learning rate','Fresh evaluation prefixes']){
  assert.ok([...document.querySelectorAll('label')].some(n=>n.textContent===label),label);
 }
 assert.ok(!document.body.textContent.includes('Supervised endpoint budget'));
 assert.equal(current.routing.value_spectral_window,8);
 assert.equal(current.routing.value_spectral_basins,1);
 assert.equal(current.routing.value_spectral_live_updates,100);
 assert.equal(current.routing.value_spectral_online_lr,.001);
 assert.equal(current.training.evaluation_interval,20);
 const snapshotToggle=[...document.querySelectorAll('label')].find(n=>n.textContent.includes('Snapshot-batch spectral learning')).querySelector('input');
 await act(()=>snapshotToggle.click());
 assert.equal(current.routing.value_spectral_snapshot,true);
 assert.equal(current.routing.value_spectral_snapshot_states,20);
 assert.equal(current.routing.value_spectral_snapshot_updates,20);
 assert.equal(current.routing.value_spectral_live_updates,1000);
 assert.equal(current.training.evaluation_interval,100);
 for(const label of ['Counterfactual states per snapshot','Fit updates per snapshot']) assert.ok([...document.querySelectorAll('label')].some(n=>n.textContent===label));
 assert.equal(current.routing.value_checkpoint_path,initial.routing.value_checkpoint_path);
 assert.match(document.body.textContent,/Planned probes: 1000/);
 await act(()=>document.querySelector('.launch').click());
 assert.equal(submitted,current); // no spurious million-token confirmation
});

test('live and saved results both expose four learning charts and trajectory outcomes',()=>{
 const audit={type:'spectral_live_audit',step:100,total:200,warmup_updates:100,phase:'bank warmup',trajectory_loss:1.2,original_loss:1.3,constant_loss:1.4,
  train_regret:.02,held_regret:.03,train_kl:.01,held_kl:.02,route_counts:[[1,2],[2,1],[1,2]]};
 for(const stored of [false,true]){
  const html=renderToStaticMarkup(React.createElement(SpectralLivePanel,{state:'completed',events:stored?[]:[audit],report:stored?{history:[audit]}:undefined,logs:[]}));
  assert.equal((html.match(/class="chart"/g)||[]).length,4);
  assert.match(html,/1\.20000/);
  assert.match(html,/1\.30000/);
  assert.match(html,/Bank warmup ends at update 100/);
  assert.doesNotMatch(html,/Perplexity/);
 }
});
