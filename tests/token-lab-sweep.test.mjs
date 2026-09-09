import {test} from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {TokenLab} from '../src/token-lab/TokenLab.tsx';
import {SweepResults,sweepDefaults} from '../src/token-lab/ValidationSweep.tsx';

test('sweep mode replaces manual family and checkpoint controls',()=>{
 const html=renderToStaticMarkup(React.createElement(TokenLab,{initialConfig:sweepDefaults}));
 for(const s of ['Specialization Validation Sweep','24 calibration runs','Number of training seeds','Leave-one-task-out','Resume previous sweep','No checkpoint paths to choose'])assert.ok(html.includes(s),s);
 for(const s of ['Common checkpoint path','Strength preset','Rerun saved specialists','Maximum prefix length'])assert.ok(!html.includes(s),s);
});

test('saved sweep reopens failures, report and resume without recomputation',()=>{
 const detail={runDirectory:'C:/sweep',report:'NULL RESULTS RETAINED',analysis:{sweep:{settings:sweepDefaults,status:'partial',planned_runs:8,
  families:{gpt:{matched_identical_control_pairs:1,mean_choice_gain:.1,verdict:'PRELIMINARY'}},
  entries:[{key:'ssm-seed-42-s100',status:'failed',error:'backend unavailable'}]}}};
 const html=renderToStaticMarkup(React.createElement(SweepResults,{detail,onResume:()=>{}}));
 for(const s of ['NULL RESULTS RETAINED','backend unavailable','Prepare resume','PRELIMINARY','0.1000'])assert.ok(html.includes(s),s);
});

test('mode switching and saved-run events preserve sweep settings and submit no stale checkpoint',async()=>{
 const {JSDOM}=await import('jsdom');
 const dom=new JSDOM('<div id="root"></div>',{url:'http://localhost'});
 const previous={window:globalThis.window,document:globalThis.document,HTMLElement:globalThis.HTMLElement,IS_REACT_ACT_ENVIRONMENT:globalThis.IS_REACT_ACT_ENVIRONMENT};
 Object.assign(globalThis,{window:dom.window,document:dom.window.document,HTMLElement:dom.window.HTMLElement,IS_REACT_ACT_ENVIRONMENT:true});
 const {createRoot}=await import('react-dom/client');
 const {mockIPC,clearMocks}=await import('@tauri-apps/api/mocks');
 const {emit}=await import('@tauri-apps/api/event');
 const base={name:'Manual',experiment_mode:'forced_specialization',families:['delta','delta'],dataset:'capability_10',topology:'independent',
  seed:42,latent_dim:64,hidden_dim:128,heads:4,sequence_length:48,window:16,train_steps:1000,batch_size:4,learning_rate:.001,
  evaluation_samples:2000,measurement_rate:1,reference_size:128,analysis_cap:2000,knn_k:4,tie_epsilon:.001,device:'cpu',
  spectral_enabled:true,deep_enabled:false,save_raw:true,spectral_rate:1,deep_rate:.05,
  common_base_source:'checkpoint',common_checkpoint:'wrong-base.pt',specialist_checkpoint:'',specialization_strength:1};
 const submitted=[];
 mockIPC((cmd,payload)=>{
  if(cmd==='get_token_lab_schema')return {defaults:base,tasks:[]};
  if(cmd==='list_token_labs')return [{run_id:'old',name:'Old 100%',status:'completed'}];
  if(cmd==='get_active_token_lab')return null;
  if(cmd==='get_token_lab')return {runId:'old',runDirectory:'C:/old',config:base,summary:{status:'completed'}};
  if(cmd==='validate_token_lab')return {valid:true};
  if(cmd==='start_token_lab'){submitted.push(payload.request.config);return {runId:'new'};}
 },{shouldMockEvents:true});
 const root=createRoot(document.getElementById('root'));
 try{
  await React.act(async()=>{root.render(React.createElement(TokenLab));});
  const mode=()=>[...document.querySelectorAll('label')].find(l=>l.textContent.startsWith('Experiment mode')).querySelector('select');
  await React.act(async()=>{mode().value='validation_sweep';mode().dispatchEvent(new window.Event('change',{bubbles:true}));});
  const saved=document.querySelector('[aria-label="Reopen saved Token Lab run"]');
  await React.act(async()=>{saved.value='old';saved.dispatchEvent(new window.Event('change',{bubbles:true}));});
  assert.equal(mode().value,'validation_sweep');
  assert.ok(!document.body.textContent.includes('Run saved-observation analysis'));
  await React.act(async()=>{[...document.querySelectorAll('button')].find(b=>b.textContent==='Run Token Lab').click();});
  assert.equal(submitted[0].experiment_mode,'validation_sweep');assert.equal(submitted[0].sweep_seed_count,3);
  assert.ok(!('common_checkpoint' in submitted[0]));
  await React.act(async()=>{await emit('token-lab-process-exit',{runId:'old',exitCode:0});});
  assert.equal(mode().value,'validation_sweep');
  await React.act(async()=>{mode().value='standard';mode().dispatchEvent(new window.Event('change',{bubbles:true}));});
  await React.act(async()=>{[...document.querySelectorAll('button')].find(b=>b.textContent==='Run Token Lab').click();});
  assert.equal(submitted[1].experiment_mode,'standard');assert.ok(!('sweep_seed_count' in submitted[1]));
 }finally{await React.act(async()=>root.unmount());clearMocks();dom.window.close();Object.assign(globalThis,previous);}
});
