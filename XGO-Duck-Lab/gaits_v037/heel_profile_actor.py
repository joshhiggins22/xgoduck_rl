"""Export a bounded, forward-gated leg correction over the verified v0.3.7 Walk actor."""
from pathlib import Path
import json
from hashlib import sha256

import numpy as np
import onnx
from onnx import helper, numpy_helper
from run_policy import _compose, _params, PREFIX, PILOT_SHA256


PARAMS = ('target_vx','lateral','yaw','stride','hip','knee','ankle','spread','pitch_bias','knee_bias','ankle_bias','lead','roll_feedback')
WALK = (.20,0,0,0,0,0,0,0,0,0,0,0,0)
RUN = (.349674,-.007446,.042372,.106181,0,0,0,0,0,0,0,0,0)
# Limits are for the search. The exported correction has a separate per-joint radian limit.
BOUNDS = ((.20,.40),(-.042,.042),(-.15,.15),(-.08,.30),(-.35,.84),(-.30,.72),
          (-.35,.72),(-.38,.42),(-.11,.11),(-.12,.12),(-.12,.12),(-.15,.75),(-.32,.32))


def export_profile(pilot, params, dest):
    if sha256(Path(pilot).read_bytes()).hexdigest() != PILOT_SHA256:
        raise ValueError('Incorrect Walk checkpoint')
    for value,(lo,hi) in zip(params,BOUNDS,strict=True):
        if not lo <= value <= hi:
            raise ValueError((value,lo,hi))
    d = dict(zip(PARAMS,params,strict=True))
    model=onnx.load(pilot)
    _compose(model,_params({k:d[k] for k in PARAMS[:4]}))
    g=model.graph
    output=g.output[0].name
    assert g.node[-1].output[0]==output
    g.node[-1].output[0]='heel_profile_base'
    def const(name, arr, dtype=np.float32):
        n='heel_profile_'+name
        g.initializer.append(numpy_helper.from_array(np.asarray(arr,dtype=dtype),n))
        return n
    def op(kind,name,*inputs,**kw):
        n='heel_profile_'+name
        g.node.append(helper.make_node(kind,list(inputs),[n],name=n,**kw))
        return n
    def legs(yaw=0,roll=0,hip=0,knee=0,ankle=0,mirror=False):
        side=[yaw,roll,hip,knee,ankle]
        return side+[0]*4+[-x if mirror else x for x in side]
    obs=g.input[0].name
    raw='heel_profile_base'
    prev=op('Gather','prev',obs,const('prev_indices',np.arange(34,48),np.int64),axis=1)
    gy=op('Gather','gravity_y',obs,const('gravity_idx',[4],np.int64),axis=1)
    offsets=const('offsets',legs(hip=d['pitch_bias'],knee=d['knee_bias'],ankle=d['ankle_bias'],mirror=True))
    scale=const('gain',legs(roll=d['spread'],hip=d['hip'],knee=d['knee'],ankle=d['ankle']))
    lead=op('Mul','anticipate',op('Sub','change',raw,prev),const('lead',legs(hip=d['lead'],knee=d['lead'],ankle=d['lead'])))
    damp=op('Mul','roll_balance',gy,const('roll_response',legs(roll=d['roll_feedback'])))
    correction=op('Sum','correction',op('Mul','excursion',raw,scale),offsets,lead,damp)
    correction=op('Clip','bounded',correction,const('min',-.19),const('max',.19))
    gated=op('Mul','gated',correction,PREFIX+'gate')
    g.node.append(helper.make_node('Add',[raw,gated],[output],name='heel_profile_output'))
    meta=model.metadata_props.add()
    meta.key,meta.value='heel_profile_fit',json.dumps(d,sort_keys=True)
    onnx.checker.check_model(model)
    onnx.save(model,dest)
