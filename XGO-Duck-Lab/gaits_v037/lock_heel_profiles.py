"""Select a stable Walk first, then choose Run at matched pace settings on both floors."""
from pathlib import Path
from shutil import copyfile
import json
import os
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from statistics import fmean
from train_heel_profiles import FEET,FLOORS,WALK,RUN,model_path,run_jobs,conditions,groups,WALK_MODEL,RUN_MODEL
from train_walk import write_json,verify_model_file


def records_for(out,heel,kind,folders,n):
    items={}
    markers=[]
    for folder in folders:
        report=folder/heel/(kind+'.json')
        if not report.is_file():
            continue
        content=json.loads(report.read_bytes())
        for index,r in enumerate(content['records'][:n]):
            params=tuple(r['params'])
            p=model_path(out,params)
            items[str(p)]=p
            if index==0:
                markers.append(str(p))
    return items,markers


def gate(rr):
    return all(r['target_reached'] and r['completed'] and r['stop_speed_mps'] is not None
               and r['stop_speed_mps']<.02 and r['stop_dist_m']<.105
               and r['max_tilt_deg']<16. and r['peak_roll_deg']<12.
               for r in rr)


def course_summary(rows):
    profiles={}
    for floor in FLOORS:
        for vx in (.15,.20):
            sample=[r for r in rows if r['case']['surface']==floor and r['case']['vx']==vx]
            profiles[f'{floor}/{vx}']=dict(speed=fmean(r['speed_mps'] for r in sample),
                roll=fmean(r['rms_roll_deg'] for r in sample),tilt=fmean(r['rms_tilt_deg'] for r in sample),
                coast=max((r['stop_dist_m'] or 0.) for r in sample),
                cross_per_m=fmean(r['max_cross_m']/max(r['forward_m'],.1) for r in sample),
                peak_tilt=max(r['max_tilt_deg'] for r in sample))
    return profiles


def choose(out,heel,kind,candidates,ref,pool,seed):
    plan=[(.20,38.,3.),(.20,90.,12.),(.20,90.,12.),(.15,43.,3.),(.15,43.,3.),(.20,4.62,None),(.15,5.82,None)]
    conditions_=conditions(heel,seed,plan)
    candidates=dict(candidates)
    candidates[str(ref)]=ref
    batches=run_jobs(pool,candidates,conditions_)
    baseline=course_summary(batches[str(ref)])
    records=[]
    for key,rr in batches.items():
        if key==str(ref) and kind=='run':
            continue
        g=course_summary(rr)
        full=[g[k]['speed']/baseline[k]['speed'] for k in g if k.endswith('/0.2')]
        touch=[g[k]['speed']/baseline[k]['speed'] for k in g if k.endswith('/0.15')]
        full_min,touch_min=min(full), min(touch)
        avg_roll=fmean(g[k]['roll'] for k in g)
        avg_tilt=fmean(g[k]['tilt'] for k in g)
        worst_tilt=max(g[k]['peak_tilt'] for k in g)
        cross=fmean(g[k]['cross_per_m'] for k in g)
        carpet_roll=fmean(g[k]['roll'] for k in g if k.startswith('residential'))
        # Prefer low roll on carpet and a straight course. Freeze Walk independently
        # before evaluating any Run model; minimum pace is the previously installed Walk.
        quality=carpet_roll+.5*avg_roll+.3*avg_tilt+.13*worst_tilt+16*cross
        if kind=='walk':
            rank=(gate(rr),full_min>=.997 and touch_min>=.985,-quality-100*max(0,1-full_min))
        else:
            margin=min(full_min,touch_min)
            # Build some margin into screening, then favor less sway/steering error.
            rank=(gate(rr),min(full_min,touch_min)>=1.39,min(full_min,touch_min)>=1.35,
                  min(margin,1.47)-.014*quality+.04*fmean(full))
        records.append(dict(model=key,hash=verify_model_file(Path(key)),rank=rank,full_gain=full_min-1,
                            touch_gain=touch_min-1,stats=g,rows=rr))
    records.sort(key=lambda r:r['rank'],reverse=True)
    winner=records[0]
    if not winner['rank'][0]:
        raise RuntimeError(f'No stable {kind} for {heel}')
    file=out/heel/(kind+'.onnx')
    file.parent.mkdir(exist_ok=True,parents=True)
    copyfile(winner['model'],file)
    write_json(out/heel/(kind+'.json'),dict(selected=winner,records=records))
    print(heel,kind,round(winner['full_gain']*100,1),round(winner['touch_gain']*100,1),
          [(k,round(v['speed'],3),round(v['roll'],2),round(v['cross_per_m'],3)) for k,v in winner['stats'].items()],flush=True)
    return file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--walk-fits',type=Path,nargs='+',required=True)
    p.add_argument('--run-fits',type=Path,nargs='+',required=True)
    p.add_argument('--heels',nargs='+',default=list(FEET),choices=FEET)
    p.add_argument('--shortlist',type=int,default=5)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--seed',type=int,default=50258000)
    args=p.parse_args()
    out=args.out.resolve()
    (out/'candidates').mkdir(exist_ok=True,parents=True)
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for heel in args.heels:
            walks,_=records_for(out,heel,'walk',args.walk_fits,args.shortlist)
            walk=choose(out,heel,'walk',walks,WALK_MODEL,pool,args.seed)
            runs,_=records_for(out,heel,'run',args.run_fits,args.shortlist)
            runs[str(RUN_MODEL)]=RUN_MODEL
            choose(out,heel,'run',runs,walk,pool,args.seed+2000)

if __name__=='__main__':
    main()
