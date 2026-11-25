"""Select Run from known failures plus new paired hardware stress and freeze six graphs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import random
import shutil
from statistics import fmean

from train_heel_profiles import FEET,FLOORS
from heel_profiles.perturb_check import apply_mismatch,mismatch
from run_audit import audit_episode
from walk_bench import Case,_environment
from train_walk import write_json,verify_model_file


def evaluate(job):
    key,path,case,spec = job
    with apply_mismatch(_environment(case.foot,case.surface),spec):
        result=audit_episode(path,case,gait=False)
    return key,result


def plan(heel,failures,phase,seed):
    if phase == 'screen':
        cases = [(Case(**{**row['case'],'foot':FEET[heel]}),row['mismatch']) for row in failures]
    else:
        cases = []
    for f,floor in enumerate(FLOORS):
        for i in range(8 if phase=='screen' else 24):
            s=seed + f + 2*i
            rng=random.Random(s)
            lo,hi=((.32,.94),(1.21,3.),(4.,6.9),(7.5,10.1))[i%4]
            duration=round(rng.uniform(lo,hi)*50)/50
            # Oversample uncalibrated full-stick operation. The former failure mode
            # appears here under physically plausible asymmetric loading.
            calibrated=i%4==3 if phase=='screen' else (i//4+i)%3==0
            c=Case(s,foot=FEET[heel],surface=floor,vx=.15 if i%5==4 else .2,
                   seconds=duration,stop=3.5,calibrated=calibrated)
            cases.append((c,mismatch(s,i)))
    return cases


def measure(rows,base):
    failures=sum(r['fell'] or not r['stop_ok'] for r in rows)
    coasts=[r['stop_peak_m'] if r['stop_peak_m'] is not None else 2. for r in rows]
    comparison={}
    for floor in FLOORS:
        idx=[i for i,r in enumerate(rows) if r['case']['surface']==floor and r['case']['seconds']>=2.]
        comparison[floor]=fmean(rows[i]['forward_m']/rows[i]['case']['seconds'] for i in idx)/max(.01,fmean(base[i]['forward_m']/base[i]['case']['seconds'] for i in idx))
    return dict(falls=sum(r['fell'] for r in rows),failures=failures,excess_coast=sum(x>.11 for x in coasts),
        gain=comparison,mean_coast=fmean(coasts),worst_coast=max(coasts),
        mean_cross=fmean(r['peak_cross_m']/max(.20,r['forward_m']) for r in rows),
        failed=[(r['case']['seed'],r['case']['surface'],r['fell_phase']) for r in rows if r['fell'] or not r['stop_ok']])


def run(pool,paths,cases):
    jobs=[(key,str(p),c,m) for key,p in paths.items() for c,m in cases]
    rows={key:[] for key in paths}
    for key,r in pool.map(evaluate,jobs,chunksize=1):
        rows[key].append(r)
    return {k:measure(rr,rows['walk']) for k,rr in rows.items() if k!='walk'}


def refine(args):
    out=args.out.resolve()
    out.mkdir(exist_ok=True,parents=True)
    source=args.source.resolve()
    history=json.loads((source/'stress.json').read_text())['episodes']
    failures=[]
    seen=set()
    for r in history:
        if not r['actors']['run']['stop_ok'] and (r['case']['seed'],r['case']['surface']) not in seen:
            failures.append(r)
            seen.add((r['case']['seed'],r['case']['surface']))
    for h in FEET:
        (out/h).mkdir(exist_ok=True)
        shutil.copyfile(source/h/'walk.onnx',out/h/'walk.onnx')
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for h in FEET:
            old=json.loads((source/h/'run-selection.json').read_bytes())['entries']
            entries={v['hash']:v for v in old if v['nominal_min']>=1.335 and v['nominal_partial']>=1.315}
            # Include cross-heel successes and randomized blends of this heel's
            # strong models. These graphs share architecture and physics, but each
            # parameterization is replayed on the actual target foot.
            import onnx
            from heel_profile_actor import PARAMS
            from train_heel_profiles import model_path,params_in_range
            import numpy as np
            rng=np.random.default_rng(args.seed+list(FEET).index(h))
            anchors=[]
            for v in entries.values():
                m=onnx.load(v['model'])
                md=next((kv.value for kv in m.metadata_props if kv.key=='heel_profile_fit'),None)
                if md:
                    p=json.loads(md)
                    anchors.append(np.array([p[k] for k in PARAMS]))
            (out/'candidates').mkdir(exist_ok=True)
            for i in range(args.extra):
                a,b=rng.integers(len(anchors),size=2);t=rng.uniform(.12,.87)
                v=(1-t)*anchors[a]+t*anchors[b]
                v[0] += rng.uniform(-.018,.004)
                v[8:11]+=rng.uniform(-.016,.016,3)
                path=model_path(out,params_in_range(v));key=verify_model_file(path)
                entries[key]=dict(hash=key,model=str(path))
            walk=source/h/'walk.onnx'
            mapping={k:Path(v['model']) for k,v in entries.items()}
            first=run(pool,{'walk':walk,**mapping},plan(h,failures,'screen',args.seed))
            def rank(v):
                g=min(v['gain'].values())
                return (v['falls'],v['failures'],int(g<1.37),v['excess_coast'],v['worst_coast'],
                        v['mean_cross']-.08*min(g,1.48))
            keys=sorted(first,key=lambda k:rank(first[k]))[:args.shortlist]
            write_json(out/h/'stress-screen.json',dict(records=first,shortlist=keys))
            print(h,'screen',[(k[:7],first[k]['failures'],round(min(first[k]['gain'].values()),2)) for k in keys],flush=True)
            second=run(pool,{'walk':walk,**{k:mapping[k] for k in keys}},plan(h,[],'confirm',args.seed+10_000))
            def total(k):
                a,b=first[k],second[k]
                g=min(min(a['gain'].values()),min(b['gain'].values()))
                return (a['falls']+b['falls'],a['failures']+b['failures'],int(g<1.37),
                        a['excess_coast']+b['excess_coast'],max(a['worst_coast'],b['worst_coast']),
                        fmean([a['mean_cross'],b['mean_cross']])-.10*min(g,1.50))
            ordered=sorted(keys,key=total)
            picked=ordered[0]
            shutil.copyfile(mapping[picked],out/h/'run.onnx')
            write_json(out/h/'run-selection.json',dict(selected=picked,source=str(mapping[picked]),scores={k:total(k) for k in ordered},entries=second))
            print(h,'locked',picked[:12],total(picked),flush=True)
    write_json(out/'frozen.json',{h:{m:verify_model_file(out/h/(m+'.onnx')) for m in ('run','walk')} for h in FEET})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('final01'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--extra',type=int,default=12)
    p.add_argument('--shortlist',type=int,default=4)
    refine(p.parse_args())
