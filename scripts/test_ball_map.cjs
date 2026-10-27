'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {position}=require('../web/ball-map.js');
const root=path.resolve(__dirname,'..'),read=name=>JSON.parse(fs.readFileSync(path.join(root,'web/data',name)));
const rink=read('overhead-20261025/replay.json'),balls=read('ball-sam-20261026/positions.json');
const at=f=>position(balls.frames[f],rink.frames[f],rink.relative_bounds);

test('visible held/airborne balls get an explicitly approximate diagram position',()=>{
  for(const f of [1575,3368,4446]){
    assert.equal(balls.frames[f].xy,null);
    const p=at(f);assert.equal(p.method,'floor_approximation');assert(p.xy.every(v=>v>=0&&v<=1));
  }
  assert(Math.abs(at(3368).xy[0]-.4581194868)<1e-8);
  assert(Math.abs(at(3368).xy[1]-.2444418144)<1e-8);
});
test('held ball uses only its identified carrier, without snapping to another player',()=>{
  const f=1478,carrier=rink.frames[f].objects.find(o=>o.track_id==='102');
  assert.deepEqual(at(f),{xy:carrier.xy,method:'carrier'});
  const fake={...balls.frames[f],carrier_track_id:'T999999'};
  assert.equal(position(fake,rink.frames[f],rink.relative_bounds).method,'floor_approximation');
});
test('camera cuts, unresolved balls and off-rink projections do not invent positions',()=>{
  assert.deepEqual(at(480),{xy:null,method:'unresolved'});
  assert.deepEqual(at(2614),{xy:null,method:'camera_unavailable'});
  assert.deepEqual(at(1132),{xy:null,method:'outside_rink'});
  const row={...rink.frames[3368],image_to_floor:[[1,0,0],[0,1,0],[0,0,0]]};
  assert.equal(position(balls.frames[3368],row,rink.relative_bounds).xy,null);
});
test('all original floor positions stay unchanged, all diagram outputs remain finite and bounded',()=>{
  const before=JSON.stringify(balls);let previous=0,added=0;
  for(const e of balls.frames){
    const p=at(e.frame);
    if(e.xy){assert.deepEqual(p,{xy:e.xy,method:'floor'});previous++;}
    else if(p.xy){assert(['carrier','floor_approximation'].includes(p.method));added++;}
    if(p.xy)assert(p.xy.length===2&&p.xy.every(v=>Number.isFinite(v)&&v>=0&&v<=1));
  }
  assert.equal(previous,1847);assert.equal(added,1475);
  assert.equal(JSON.stringify(balls),before);
});
