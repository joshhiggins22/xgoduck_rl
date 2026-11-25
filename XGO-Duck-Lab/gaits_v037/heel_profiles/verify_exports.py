"""Check each selected graph through the 0.3.7 runtime and its unchanged command paths."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from walk_bench import _actor
from train_walk import write_json

def check(root,pilot):
    ref=_actor(pilot.resolve())
    rng=np.random.default_rng(582201)
    inert=[(0,0,0),(-.12,0,0),(.2,.15,0),(.2,0,.7)]
    active=[(.2,0,0),(.15,0,0),(.08,.01,.02),(.17,0,-.08)]
    examples=[]
    for cmd in inert+active:
        for _ in range(8):
            x=rng.normal(0,.28,(1,61)).astype(np.float32)
            x[0,5]-=1.
            x[0,48:51]=cmd
            examples.append((cmd,x))
    hashes={}
    for h in ('5mm','factory','10mm'):
        hashes[h]={}
        for mode in ('walk','run'):
            p=(root/h/(mode+'.onnx')).resolve(strict=True)
            actor=_actor(p)
            assert actor.loaded and actor.warm(12).shape==(1,14)
            np.testing.assert_array_equal(actor.home_deg,ref.home_deg)
            deltas=[]
            for cmd,x in examples:
                a=actor.session.run(None,{actor.in_name:x})[0]
                b=ref.session.run(None,{ref.in_name:x})[0]
                assert a.shape==(1,14) and np.isfinite(a).all()
                if cmd in inert: np.testing.assert_array_equal(a,b)
                else: deltas.append(float(np.max(np.abs(a-b))))
            hashes[h][mode]=dict(sha256=sha256(p.read_bytes()).hexdigest(),samples=len(examples),max_forward_difference_rad=max(deltas))
    return hashes
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('models',type=Path);p.add_argument('reference',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args()
    write_json(a.out,check(a.models,a.reference))
