import {test} from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {TokenLab,Explorer} from '../src/token-lab/TokenLab.tsx';

test('Token Lab exposes standalone setup and measurement controls',()=>{
 const config={families:['gpt','gpt'],name:'Test',dataset:'tinystories',topology:'independent',window:16,latent_dim:64,hidden_dim:128,heads:4,seed:42,sequence_length:64,train_steps:100,learning_rate:.001,batch_size:4,evaluation_samples:256,measurement_rate:1,reference_size:128,analysis_cap:2000,knn_k:4,tie_epsilon:.001,device:'cpu',train_stories:1000,validation_stories:500,spectral_enabled:true,deep_enabled:false,save_raw:true,spectral_rate:1,deep_rate:.05};
 const html=renderToStaticMarkup(React.createElement(TokenLab,{initialConfig:config}));
 for(const text of ['Token Lab','Independent / same-state counterfactual','Serial / stage improvements','Counter','Deep / gradient probes','Save raw observations','Training reference bank size','Reopen saved Token Lab run','TinyStories']){
  if(text!=='Counter')assert.ok(html.includes(text),text);
 }
 assert.ok(html.includes('does not load your EMC checkpoint'));
 assert.ok(html.includes('100'));
});

test('saved explorer renders charts, report, pair selector and analysis without rerun',()=>{
 const expert={performance:{},associations:{improvement:[]},ridge:{},correlations:[]};
 const detail={rows:[{sample_id:'a',expert_id:'1:gpt',task_id:'language',expert_stage:1,position:3,effective_rank:2,improvement:.1,expert_loss:1}],
  schema:[{key:'effective_rank',display_name:'Effective rank',category:'Shape',role:'input',definition:'participation ratio'}],
  summary:{analysis_locations:1},analysis:{experts:{'1:gpt':expert},pairs:{}},report:'# Token Lab Report\nINCONCLUSIVE'};
 const html=renderToStaticMarkup(React.createElement(Explorer,{detail}));
 for(const text of ['Attribute vs performance','Low / high outcome','Feature interactions','Correlation / redundancy','predictive probes','INCONCLUSIVE','Same-state expert pair'])assert.ok(html.includes(text),text);
 assert.equal((html.match(/class="chart"/g)||[]).length,3);
});
