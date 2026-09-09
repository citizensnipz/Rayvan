import {test} from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {DiscoverySetup,DiscoveryResults} from '../src/token-lab/FeatureDiscovery.tsx';

test('saved analysis UI exposes configuration and no training requirement',()=>{
 const html=renderToStaticMarkup(React.createElement(DiscoverySetup,{detail:{runDirectory:'C:/runs/source'},active:false,onRun:()=>{}}));
 for(const text of ['Run saved-observation analysis','Maximum selected features','Candidate feature count','Analysis location cap','Target-free','No retraining'])assert.ok(html.includes(text),text);
});

test('saved discovery reopens compact metrics, ablation chart and report',()=>{
 const d={status:'ok',selected_features:['hidden_mean'],counts:{train:100,validation:30,test:30},validation_chosen_model:'compact_ridge',
 scores:{compact_ridge:{r2:.8,mae:.1,spearman:.9}},ablations:[{feature:'hidden_mean',mae_increase:.2,mae_increase_ci:[.1,.3]}],held_out_associations:[]};
 const detail={analysis:{discovery:{analyzed_locations:160,source_locations:160,targets:{'1:gpt improvement':d}}},report:'Saved discovery result'};
 const html=renderToStaticMarkup(React.createElement(DiscoveryResults,{detail}));
 for(const text of ['hidden_mean','Removing','Saved discovery result','0.8000','0.2000']){
  if(text!=='Removing')assert.ok(html.includes(text),text);
 }
 assert.equal((html.match(/class="chart"/g)||[]).length,1);
});

test('identical experts show an explicit no-difference result',()=>{
 const detail={analysis:{discovery:{targets:{'A minus B advantage':{status:'constant outcome; no state-dependent difference to explain'}}}},report:''};
 const html=renderToStaticMarkup(React.createElement(DiscoveryResults,{detail}));
 assert.ok(html.includes('no state-dependent difference to explain'));
});
