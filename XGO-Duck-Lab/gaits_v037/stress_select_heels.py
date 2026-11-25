"""Evaluate gait finalists under unequal leg strength and IMU/encoder offsets before freezing."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import shutil
from statistics import fmean
from train_heel_profiles import FEET,FLOORS,WALK_MODEL
from heel_profiles.perturb_check import apply_mismatch,mismatch
from walk_bench import Case,_environment
from run_audit import audit_episode
from train_walk import write_json,verify_model_file

def evaluate(job):
    path,case,i,distance=job
    plant=_environment(case.foot,case.surface)
    with apply_mismatch(plant,mismatch(case.seed,i)):
        return audit_episode(path,case,distance,gait=False)

def rank(kind,nominal,rr,reference,walk_nominal):
    groups={}
    failures=0
    violations=0
    for floor in FLOORS:
        for vx in (.15,.2):
            idx=[i for i,r in enumerate(rr) if (r['case']['surface'],r['case']['vx'])==(floor,vx)]
            rows=[rr[i] for i in idx];base=[reference[i] for i in idx]
            ratios=[]
            for r,b in zip(rows,base,strict=True):
                if r['fell'] or not r['stop_ok'] or r['reached'] is False:
                    failures+=1
                if (r['max_tilt_deg']>16 or r['peak_roll_deg']>12 or (r['stop_peak_m'] or 0)>.11
                    or (r['settle_s'] is not None and r['settle_s']>1.0)):
                    violations+=1
                if r['distance_target_m'] is not None and r['finish_s'] and b['finish_s']:
                    ratios.append(b['finish_s']/r['finish_s'])
            k=f'{floor}/{vx}'
            groups[k]=dict(gain=fmean(ratios) if ratios else 0.,
                roll=fmean(r['rms_roll_deg'] for r in rows),tilt=fmean(r['rms_tilt_deg'] for r in rows),
                drift=fmean(r['peak_cross_m']/max(.20,r['forward_m']) for r in rows))
    stable=[v for k,v in groups.items() if k.startswith('residential')]
    cost=fmean(x['roll']+.31*x['tilt']+9*x['drift'] for x in groups.values())+.4*fmean(x['roll'] for x in stable)
    full=[k for k in groups if k.endswith('/0.2')]
    adjusted=min(groups[k]['gain'] for k in full)
    nominal_ratios={k:v['speed']/walk_nominal[k]['speed'] for k,v in nominal['stats'].items()}
    nominal_min=min(nominal_ratios[k] for k in full)
    nominal_partial=min(nominal_ratios.values())
    base_gate=nominal['rank'][0]
    if kind=='walk':
        score=(base_gate,-failures,-violations,nominal['rank'][1],adjusted>=.98,-cost)
    else:
        score=(base_gate,-failures,-violations,nominal_min>=1.38 and nominal_partial>=1.35,
               adjusted>=1.35,
               -cost+.6*min(adjusted,1.55)+.6*min(nominal_min,1.52)
               -20*max(0.,1.38-nominal_min)-15*max(0.,1.35-nominal_partial)-12*max(0.,1.35-adjusted))
    return dict(score=score,nominal_min=nominal_min,nominal_partial=nominal_partial,stress_min=adjusted,failures=failures,violations=violations,groups=groups)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--heel',choices=FEET,required=True)
    p.add_argument('--workers',type=int,default=5)
    args=p.parse_args();heel=args.heel
    base=args.prior/heel
    out=args.out/heel
    out.mkdir(parents=True,exist_ok=True)
    seed=27119900
    plan=[(.2,45,3.),(.2,45,3.),(.15,47,3.),(.15,47,3.),(.2,7.16,None),(.15,4.44,None),(.2,.62,None)]
    cases=[(Case(seed+2*i+f,foot=FEET[heel],surface=floor,vx=v,seconds=t,stop=3.5,calibrated=i%5!=3),i,d)
          for f,floor in enumerate(FLOORS) for i,(v,t,d) in enumerate(plan)]
    memo={}
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        def rows(path):
            key=verify_model_file(Path(path))
            if key not in memo:
                memo[key]=list(pool.map(evaluate,[(path,c,i,d) for c,i,d in cases]))
            return memo[key]
        for mode in ('walk','run'):
            records=json.loads((base/(mode+'.json')).read_text())['records']
            records=records[:5] if mode=='walk' else records[:10]
            baseline=WALK_MODEL if mode=='walk' else out/'walk.onnx'
            base_rows=rows(baseline)
            ranked=[]
            for r in records:
                rr=rows(r['model'])
                guide=r['stats'] if mode=='walk' else locked_walk
                stats=rank(mode,r,rr,base_rows,guide)
                ranked.append(dict(model=r['model'],hash=r['hash'],**stats,nominal=r['stats'],episodes=rr))
            ranked.sort(key=lambda r:r['score'],reverse=True)
            win=ranked[0]
            if mode=='walk': locked_walk=win['nominal']
            shutil.copyfile(win['model'],out/(mode+'.onnx'))
            write_json(out/(mode+'-selection.json'),dict(selected=win['hash'],entries=ranked))
            print(heel,mode,win['hash'][:10],win['failures'],win['violations'],round(win['nominal_min'],3),round(win['stress_min'],3),flush=True)
if __name__=='__main__':
    main()
