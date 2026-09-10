import {test} from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {TokenLab} from '../src/token-lab/TokenLab.tsx';
import {SweepResults,sweepDefaults} from '../src/token-lab/ValidationSweep.tsx';
import {emergenceDefaults} from '../src/token-lab/EmergentSpecialization.tsx';

test('mixed sweep shows one shared four-expert bank, six pairs and exposure control',()=>{
 const html=renderToStaticMarkup(React.createElement(TokenLab,{initialConfig:{...sweepDefaults,population:'heterogeneous',sweep_seed_count:4}}));
 for(const s of ['8 calibration runs','One mixed bank','all six cross-family','not identical experts','four seeds','Sweep population'])assert.ok(html.includes(s),s);
});

test('mixed result offers emergence and all six pairs, not unsupported frozen pair routing',()=>{
 const detail={analysis:{sweep:{settings:{population:'heterogeneous'},entries:[],families:{},status:'completed',planned_runs:8}}};
 const html=renderToStaticMarkup(React.createElement(SweepResults,{detail,onResume:()=>{},onRouting:()=>{},onEmergence:()=>{}}));
 assert.ok(html.includes('Test emergence with this general bank'));
 assert.ok(html.includes('All six pair comparisons'));
 assert.ok(!html.includes('Test routing with these saved experts'));
});

test('mixed emergence explicitly labels nonidentical baseline and fixes population at four',()=>{
 const html=renderToStaticMarkup(React.createElement(TokenLab,{initialConfig:{...emergenceDefaults,family:'mixed'}}));
 for(const s of ['jointly pretrained GPT/SSM/GRU/Delta','mixed equal-exposure baseline','Cross-family parameter distances are unavailable','Expert count<select disabled'])assert.ok(html.includes(s),s);
});

test('mixed bank button and smoke preset retain all four architectures in submitted request',async()=>{
 const {JSDOM}=await import('jsdom');const dom=new JSDOM('<div id="root"></div>',{url:'http://localhost'});
 const old={window:globalThis.window,document:globalThis.document,HTMLElement:globalThis.HTMLElement,ResizeObserver:globalThis.ResizeObserver,IS_REACT_ACT_ENVIRONMENT:globalThis.IS_REACT_ACT_ENVIRONMENT};
 Object.assign(globalThis,{window:dom.window,document:dom.window.document,HTMLElement:dom.window.HTMLElement,ResizeObserver:class {observe(){} disconnect(){}},IS_REACT_ACT_ENVIRONMENT:true});
 Object.defineProperty(dom.window.HTMLElement.prototype,'clientWidth',{get:()=>800});Object.defineProperty(dom.window.HTMLElement.prototype,'clientHeight',{get:()=>300});
 dom.window.HTMLCanvasElement.prototype.getContext=function(){return new Proxy({canvas:this,measureText:t=>({width:String(t).length*7}),createLinearGradient:()=>({addColorStop(){}})},{get:(o,k)=>k in o?o[k]:(()=>{})});};
 const {createRoot}=await import('react-dom/client');const {mockIPC,clearMocks}=await import('@tauri-apps/api/mocks');
 const settings={...sweepDefaults,population:'heterogeneous',sweep_seed_count:4};const submitted=[];
 mockIPC((cmd,p)=>{
  if(cmd==='get_token_lab_schema')return {defaults:sweepDefaults,tasks:[]};
  if(cmd==='list_token_labs')return [{run_id:'mixed',name:'Mixed',status:'completed',experiment_type:'token_lab_sweep',runDirectory:'C:/mixed'}];
  if(cmd==='get_active_token_lab')return null;
  if(cmd==='get_token_lab')return {runId:'mixed',runDirectory:'C:/mixed',config:settings,summary:{status:'completed'},analysis:{sweep:{settings,entries:[42,43,44,45].map(seed=>({key:`mixed-${seed}`,seed,family:'mixed',strength:1,status:'completed'})),families:{},planned_runs:8}}};
  if(cmd==='validate_token_lab')return {valid:true};
  if(cmd==='start_token_lab'){submitted.push(p.request.config);return {runId:'new'};}
 },{shouldMockEvents:true});
 const root=createRoot(document.getElementById('root'));
 try{
  await React.act(async()=>root.render(React.createElement(TokenLab)));
  await React.act(async()=>{const select=document.querySelector('[aria-label="Reopen saved Token Lab run"]');select.value='mixed';select.dispatchEvent(new window.Event('change',{bubbles:true}));});
  const click=async label=>React.act(async()=>[...document.querySelectorAll('button')].find(b=>b.textContent===label).click());
  await click('Test emergence with this general bank');await click('Smoke test preset');await click('Run Token Lab');
  assert.equal(submitted[0].family,'mixed');assert.equal(submitted[0].expert_count,4);assert.equal(submitted[0].source_sweep,'C:/mixed');assert.equal(submitted[0].experiment_mode,'emergent_specialization');
  assert.ok(!('task_weights' in submitted[0]));
 }finally{await React.act(async()=>root.unmount());clearMocks();dom.window.close();Object.assign(globalThis,old);}
});
