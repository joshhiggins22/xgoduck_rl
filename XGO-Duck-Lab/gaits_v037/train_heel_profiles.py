"""Fit six ONNX gait profiles across hardwood and residential carpet on paired courses."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
from statistics import fmean

for var in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[var]='1'

import numpy as np
from heel_profile_actor import PARAMS, BOUNDS, WALK, RUN, export_profile
from train_walk import write_json
from walk_bench import Case, simulate

FLOORS=('hardwood','residential-carpet')
FEET={'factory':'v2-stock','5mm':'v2-heel5','10mm':'v2-heel10'}
ROOT=Path(__file__).resolve().parents[1] / 'runs'
WALK_MODEL=ROOT/'v037-v2-heel5-forward-walk'/'walk.onnx'
RUN_MODEL=ROOT/'v037-v2-heel5-forward-run'/'xgoduck_run.onnx'


def params_in_range(values):
    return tuple(round(float(np.clip(v,*bound)),6) for v,bound in zip(values,BOUNDS,strict=True))


def candidates(rng, kind, count, anchors, radius):
    base=np.array(WALK if kind=='walk' else RUN)
    sig=np.array((.030,.012,.045,.115,.24,.18,.22,.15,.041,.041,.043,.24,.13))
    if kind=='walk':
        sig *= .65
    points=[params_in_range(p) for p in anchors]
    if radius==1:
        for j in range(3,len(base)):
            for sign in (-1,1):
                p=base.copy()
                p[j]+=sign*sig[j]
                points.append(params_in_range(p))
    while len(set(points))<count:
        p=np.array(anchors[rng.integers(len(anchors))])
        points.append(params_in_range(p+rng.normal(size=len(p))*sig*radius))
    return list(dict.fromkeys(points))


def model_path(out, params):
    if tuple(params)==WALK:
        return WALK_MODEL
    if tuple(params)==RUN:
        return RUN_MODEL
    path=(out/'candidates'/ (sha256(json.dumps(params).encode()).hexdigest()[:18]+'.onnx')).resolve()
    if not path.is_file():
        export_profile(WALK_MODEL,params,path)
    return path


def episode(job):
    label,model,case,distance=job
    row=simulate(None,case,model,finish_forward_m=distance)
    row['target_reached'] = distance is None or row['forward_m'] >= distance - 1e-8
    return label,row


def conditions(foot,seed,plan):
    return [(Case(seed+idx*2+i,foot=FEET[foot],surface=floor,vx=v,seconds=t,stop=2.5),dist)
            for i,floor in enumerate(FLOORS) for idx,(v,t,dist) in enumerate(plan)]


def run_jobs(pool,mapping,conditions_):
    by_model=defaultdict(list)
    jobs=[(label,path,c,d) for label,path in mapping.items() for c,d in conditions_]
    for label,row in pool.map(episode,jobs,chunksize=2):
        by_model[label].append(row)
    return by_model


def avg(rows,k):
    return fmean(r[k] for r in rows)


def groups(rows):
    bucket=defaultdict(list)
    for r in rows:
        bucket[(r['case']['surface'],r['case']['vx'])].append(r)
    result={}
    for (floor,v),rr in bucket.items():
        completed=[x for x in rr if x['completed'] and x['stop_speed_mps'] is not None]
        result[f'{floor}/{v}']=dict(
            speed=avg(rr,'speed_mps'), roll=avg(rr,'rms_roll_deg'), tilt=avg(rr,'rms_tilt_deg'),
            peakroll=max(r['peak_roll_deg'] for r in rr), peaktilt=max(r['max_tilt_deg'] for r in rr),
            drift=avg(rr,'rms_cross_m'), peak_drift=avg(rr,'max_cross_m'),
            failures=sum(not r['target_reached'] or not r['completed'] or r['stop_speed_mps'] is None or r['stop_speed_mps']>.02 for r in rr),
            coast=max((r['stop_dist_m'] for r in completed),default=10.))
    return result


def rank(kind,g,ref):
    fail=sum(m['failures'] for m in g.values())
    full=[k for k in g if k.endswith('/0.2')]
    ratios=[g[k]['speed']/max(ref[k]['speed'],.01) for k in full]
    # Run's target is measured separately on each floor; reward margin for held-out starts.
    # Walk is required to retain reference pace, then scored on balance and line keeping.
    good=all(m['peaktilt']<=16 and m['peakroll']<=13 and m['coast']<=.105 for m in g.values())
    wobble=fmean(m['roll']+.35*m['tilt'] for m in g.values())
    side=fmean(m['drift'] for m in g.values())
    if kind=='walk':
        floor=min(g[k]['speed']/max(ref[k]['speed'],.01) for k in g)
        stable_utility=-wobble-14*side-.2*max(m['peakroll'] for m in g.values())
        return (-fail,int(good),int(floor>=.995),stable_utility-80*max(0.,.995-floor))
    minimum=min(ratios)
    score=.75*min(minimum,1.52)+.25*fmean(min(v,1.56) for v in ratios)-.011*wobble-.38*side
    return (-fail,int(good),int(minimum>=1.43),score)


def tournament(pool,out,heel,kind,points,seed,plan,baseline):
    points=list(dict.fromkeys(points))
    mapping={str(i):model_path(out,p) for i,p in enumerate(points)}
    mapping['reference']=baseline
    result=run_jobs(pool,mapping,conditions(heel,seed,plan))
    ref=groups(result.pop('reference'))
    records=[]
    for idx,rows in result.items():
        g=groups(rows)
        records.append(dict(params=points[int(idx)],groups=g,rank=rank(kind,g,ref)))
    records.sort(key=lambda r:r['rank'],reverse=True)
    return dict(reference=ref,records=records)


def train(args):
    out=args.out.resolve()
    (out/'candidates').mkdir(exist_ok=True,parents=True)
    rng=np.random.default_rng(args.seed)
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for kind in args.kinds:
            start=WALK if kind=='walk' else RUN
            for j,heel in enumerate(args.heels):
                dest=out/heel
                dest.mkdir(exist_ok=True)
                baseline=WALK_MODEL if kind=='walk' else dest/'walk.onnx'
                assert baseline.is_file(),baseline
                seed=args.seed+2500*j+(10000 if kind=='run' else 0)
                anchors=[start]
                if args.prior:
                    last=json.loads((args.prior/heel/(kind+'.json')).read_text())
                    anchors.extend(tuple(r['params']) for r in last['records'][:4])
                    for other in FEET:
                        if other != heel and (args.prior/other/(kind+'.json')).is_file():
                            rows=json.loads((args.prior/other/(kind+'.json')).read_text())['records']
                            anchors.extend(tuple(r['params']) for r in rows[:2])
                points=candidates(rng,kind,args.population,anchors,1.)
                short=[(.2,4.2,None)]
                first=tournament(pool,out,heel,kind,points,seed,short,baseline)
                write_json(dest/(kind+'-screen.json'),first)
                top=[tuple(r['params']) for r in first['records'][:6]]
                newpoints=candidates(rng,kind,args.refine,top,.55)
                second=tournament(pool,out,heel,kind,newpoints+[start],seed+300,[(.2,6.24,None),(.15,4.7,None)],baseline)
                write_json(dest/(kind+'-refine.json'),second)
                finalists=list(dict.fromkeys([start]+top[:3]+[tuple(r['params']) for r in second['records'][:7]]))
                plan=[(.2,48.,3.),(.2,48.,3.),(.2,6.56,None),(.15,5.38,None)]
                final=tournament(pool,out,heel,kind,finalists,seed+600,plan,baseline)
                write_json(dest/(kind+'.json'),final)
                params=final['records'][0]['params']
                target=dest/(kind+'.onnx')
                target.write_bytes(model_path(out,params).read_bytes())
                print(heel,kind,[(k,round(g['speed'],3),round(g['roll'],2),g['failures']) for k,g in final['records'][0]['groups'].items()],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prior',type=Path)
    p.add_argument('--seed',type=int,default=521125)
    p.add_argument('--population',type=int,default=100)
    p.add_argument('--refine',type=int,default=52)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--kinds',nargs='+',choices=('walk','run'),default=['walk','run'])
    p.add_argument('--heels',nargs='+',choices=FEET,default=['5mm','factory','10mm'])
    train(p.parse_args())
