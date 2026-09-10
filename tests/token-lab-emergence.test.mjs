import {test} from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {TokenLab} from '../src/token-lab/TokenLab.tsx';
import {emergenceDefaults,EmergenceResults} from '../src/token-lab/EmergentSpecialization.tsx';
import {sweepDefaults} from '../src/token-lab/ValidationSweep.tsx';

test('emergence mode exposes controlled presets instead of specialist task settings',()=>{
 const html=renderToStaticMarkup(React.createElement(TokenLab,{initialConfig:emergenceDefaults}));
 for(const s of ['Emergent Specialization','Smoke test preset','Narrow study preset','42 condition/seed jobs','Asymmetry strengths','Competence feedback strengths','pre-specialization common-base.pt','Matched held-out utility states'])assert.ok(html.includes(s),s);
 for(const s of ['Specialization strength','Task Distribution','Expert hidden dimension','NaN','Training optimizer steps'])assert.ok(!html.includes(s),s);
});

test('partial saved emergence report reopens with failures and separate hypotheses',()=>{
 const detail={analysis:{emergence:{status:'partial',entries:[{seed:42,id:'D',status:'failed',error:'missing common base'}],planned_jobs:5,summary:{}}},report:'NULL FINDINGS PRESERVED'};
 const html=renderToStaticMarkup(React.createElement(EmergenceResults,{detail}));
 for(const s of ['Emergence','Detectability','Utility','H1 emergence','H2 detection','H3 utility','missing common base','NULL FINDINGS PRESERVED'])assert.ok(html.includes(s),s);
});

test('UI mode, source, preset and IPC; reopening results cannot overwrite the chosen base',async()=>{
 const {JSDOM}=await import('jsdom');const dom=new JSDOM('<div id="root"></div>',{url:'http://localhost'});
 const old={window:globalThis.window,document:globalThis.document,HTMLElement:globalThis.HTMLElement,ResizeObserver:globalThis.ResizeObserver,IS_REACT_ACT_ENVIRONMENT:globalThis.IS_REACT_ACT_ENVIRONMENT};
 Object.assign(globalThis,{window:dom.window,document:dom.window.document,HTMLElement:dom.window.HTMLElement,ResizeObserver:class {observe(){} disconnect(){}},IS_REACT_ACT_ENVIRONMENT:true});
 Object.defineProperty(dom.window.HTMLElement.prototype,'clientWidth',{get:()=>800});Object.defineProperty(dom.window.HTMLElement.prototype,'clientHeight',{get:()=>300});
 dom.window.HTMLCanvasElement.prototype.getContext=function(){return new Proxy({canvas:this,measureText:t=>({width:String(t).length*7}),createLinearGradient:()=>({addColorStop(){}})},{get:(o,k)=>k in o?o[k]:(()=>{})});};
 const {createRoot}=await import('react-dom/client');const {mockIPC,clearMocks}=await import('@tauri-apps/api/mocks');
 const submitted=[];
 const entries=[42,43,44].map(seed=>({key:`gpt-${seed}`,seed,family:'gpt',strength:1,status:'completed'}));
 mockIPC((cmd,p)=>{
  if(cmd==='get_token_lab_schema')return {defaults:sweepDefaults,tasks:[]};
  if(cmd==='list_token_labs')return ['a','b'].map(id=>({run_id:id,name:id,status:'completed',experiment_type:'token_lab_sweep',runDirectory:`C:/${id}`}));
  if(cmd==='get_active_token_lab')return null;
  if(cmd==='get_token_lab')return {runId:p.runId,runDirectory:`C:/${p.runId}`,config:sweepDefaults,summary:{status:'completed'},analysis:{sweep:{settings:sweepDefaults,status:'completed',entries,planned_runs:3,families:{}}}};
  if(cmd==='validate_token_lab')return {valid:true};
  if(cmd==='start_token_lab'){submitted.push(p.request.config);return {runId:'new'};}
 },{shouldMockEvents:true});
 const root=createRoot(document.getElementById('root'));
 const change=async(s,value)=>React.act(async()=>{s.value=value;s.dispatchEvent(new window.Event('change',{bubbles:true}));});
 try{
  await React.act(async()=>root.render(React.createElement(TokenLab)));
  await change([...document.querySelectorAll('select')].find(s=>s.parentElement.textContent.startsWith('Experiment mode')),'emergent_specialization');
  await change(document.querySelector('[aria-label="Emergence source validation sweep"]'),'a');
  await React.act(async()=>[...document.querySelectorAll('button')].find(b=>b.textContent==='Smoke test preset').click());
  await change(document.querySelector('[aria-label="Reopen saved Token Lab run"]'),'b');
  assert.equal(document.querySelector('[aria-label="Emergence source validation sweep"]').value,'a');
  await React.act(async()=>[...document.querySelectorAll('button')].find(b=>b.textContent==='Run Token Lab').click());
  const c=submitted[0];assert.equal(c.experiment_mode,'emergent_specialization');assert.equal(c.source_sweep,'C:/a');assert.deepEqual(c.seeds,[42]);assert.equal(c.expert_count,2);assert.equal(c.train_steps,10);
  for(const k of ['specialist_checkpoint','specialization_strength','task_weights','families'])assert.ok(!(k in c));
 }finally{await React.act(async()=>root.unmount());clearMocks();dom.window.close();Object.assign(globalThis,old);}
});
