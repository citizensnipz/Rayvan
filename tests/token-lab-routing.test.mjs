import {test} from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {TokenLab} from '../src/token-lab/TokenLab.tsx';
import {RoutingResults,routingDefaults} from '../src/token-lab/FrozenRouting.tsx';
import {SweepResults,sweepDefaults} from '../src/token-lab/ValidationSweep.tsx';

test('routing setup replaces training/checkpoint controls with saved-sweep selector',()=>{
 const html=renderToStaticMarkup(React.createElement(TokenLab,{initialConfig:routingDefaults}));
 for(const s of ['Frozen Expert Routing Test','Source validation sweep','Include paired 0%','one step and 3 sequential steps','Fresh router training locations'])assert.ok(html.includes(s),s);
 for(const s of ['Common checkpoint path','Expert hidden dimension','Training optimizer steps','NaN'])assert.ok(!html.includes(s),s);
});

test('routing results reopen without ordinary explorer or training',()=>{
 const detail={analysis:{routing_test:{status:'partial',entries:[{key:'gpt-42',status:'failed',error:'missing model.pt'}],planned_jobs:1}},report:'ALL NULL RESULTS RETAINED'};
 const html=renderToStaticMarkup(React.createElement(RoutingResults,{detail}));
 for(const s of ['ALL NULL RESULTS RETAINED','missing model.pt','Context tok/s','Suffix regret'])assert.ok(html.includes(s),s);
 const sweep={runDirectory:'C:/sweep',analysis:{sweep:{settings:sweepDefaults,status:'completed',planned_runs:0,entries:[],families:{}}}};
 assert.ok(renderToStaticMarkup(React.createElement(SweepResults,{detail:sweep,onResume:()=>{},onRouting:()=>{}})).includes('Test routing with these saved experts'));
});

test('sweep button chooses its source; reopening another run never overwrites routing settings',async()=>{
 const {JSDOM}=await import('jsdom');const dom=new JSDOM('<div id="root"></div>',{url:'http://localhost'});
 const previous={window:globalThis.window,document:globalThis.document,HTMLElement:globalThis.HTMLElement,ResizeObserver:globalThis.ResizeObserver,IS_REACT_ACT_ENVIRONMENT:globalThis.IS_REACT_ACT_ENVIRONMENT};
 Object.assign(globalThis,{window:dom.window,document:dom.window.document,HTMLElement:dom.window.HTMLElement,ResizeObserver:class {observe(){} disconnect(){}},IS_REACT_ACT_ENVIRONMENT:true});
 // Canvas is a rendering stub; this test checks form state and IPC, not pixels.
 Object.defineProperty(dom.window.HTMLElement.prototype,'clientWidth',{get:()=>800});
 Object.defineProperty(dom.window.HTMLElement.prototype,'clientHeight',{get:()=>300});
 dom.window.HTMLCanvasElement.prototype.getContext=function(){return new Proxy({canvas:this,measureText:t=>({width:String(t).length*7}),createLinearGradient:()=>({addColorStop(){}})}, {get:(o,k)=>k in o?o[k]:(()=>{})});};
 const {createRoot}=await import('react-dom/client');const {mockIPC,clearMocks}=await import('@tauri-apps/api/mocks');
 const detail=id=>({runId:id,runDirectory:`C:/${id}`,config:sweepDefaults,summary:{status:'completed'},analysis:{sweep:{settings:sweepDefaults,status:'completed',planned_runs:0,entries:[],families:{}}}});
 const submitted=[];
 mockIPC((cmd,p)=>{
  if(cmd==='get_token_lab_schema')return {defaults:sweepDefaults,tasks:[]};
  if(cmd==='list_token_labs')return ['sweep-a','sweep-b'].map(id=>({run_id:id,name:id,status:'completed',experiment_type:'token_lab_sweep',runDirectory:`C:/${id}`}));
  if(cmd==='get_active_token_lab')return null;
  if(cmd==='get_token_lab')return detail(p.runId);
  if(cmd==='validate_token_lab')return {valid:true};
  if(cmd==='start_token_lab'){submitted.push(p.request.config);return {runId:'new'};}
 },{shouldMockEvents:true});
 const root=createRoot(document.getElementById('root'));
 try{
  await React.act(async()=>root.render(React.createElement(TokenLab)));
  const open=async id=>{await React.act(async()=>{const s=document.querySelector('[aria-label="Reopen saved Token Lab run"]');s.value=id;s.dispatchEvent(new window.Event('change',{bubbles:true}));});};
  await open('sweep-a');
  await React.act(async()=>[...document.querySelectorAll('button')].find(b=>b.textContent==='Test routing with these saved experts').click());
  assert.equal(document.querySelector('[aria-label="Source validation sweep"]').value,'sweep-a');
  await open('sweep-b');
  assert.equal(document.querySelector('[aria-label="Source validation sweep"]').value,'sweep-a');
  await React.act(async()=>[...document.querySelectorAll('button')].find(b=>b.textContent==='Run Token Lab').click());
  assert.equal(submitted[0].source_sweep,'C:/sweep-a');assert.equal(submitted[0].experiment_mode,'frozen_routing');
  for(const k of ['common_checkpoint','specialist_checkpoint','families','train_steps'])assert.ok(!(k in submitted[0]));
 }finally{await React.act(async()=>root.unmount());clearMocks();dom.window.close();Object.assign(globalThis,previous);}
});
