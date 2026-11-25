"""Re-run each Walk/Run pair on unseen matched starts, releases and endurance courses."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from statistics import fmean

for var in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[var]='1'
from walk_bench import Case
from run_audit import audit_episode
from train_walk import write_json

FEET={'factory':'v2-stock','5mm':'v2-heel5','10mm':'v2-heel10'}
FLOORS=('hardwood','residential-carpet')

def one(job):
    heel,profile,model,case,distance,group=job
    row=audit_episode(model,case,distance,gait=False)
    return heel,case.surface,group,profile,row

def describe(rows):
    done=[r for r in rows if r['finish_s'] is not None]
    result=dict(trials=len(rows),falls=sum(r['fell'] for r in rows),stops=sum(r['stop_ok'] for r in rows),
        completed=sum(r['reached'] for r in rows if r['reached'] is not None),
        rms_roll=fmean(r['rms_roll_deg'] for r in rows), rms_tilt=fmean(r['rms_tilt_deg'] for r in rows),
        peak_roll=max(r['peak_roll_deg'] for r in rows),peak_tilt=max(r['max_tilt_deg'] for r in rows),
        mean_peak_drift=fmean(r['peak_cross_m'] for r in rows),worst_peak_drift=max(r['peak_cross_m'] for r in rows))
    for key,meas in [('worst_coast','stop_peak_m'),('worst_settle','settle_s')]:
        values=[r[meas] for r in rows if r[meas] is not None]
        result[key]=max(values,default=None)
    result['speed_mps']=(fmean(r['distance_target_m']/r['finish_s'] for r in done)
        if done else fmean(r['forward_m']/max(.02,r['commanded_s']) for r in rows))
    result['time_s']=fmean(r['finish_s'] for r in done) if done else None
    return result

def main(args):
    folder=args.source.resolve()
    models={h:{g:(folder/('installed-'+g+'.onnx') if args.installed else folder/h/(g+'.onnx')) for g in ('walk','run')} for h in args.heels}
    digests={h:{g:hashlib.sha256(p.read_bytes()).hexdigest() for g,p in pair.items()} for h,pair in models.items()}
    jobs=[]
    # Drive input is mirrored across heels, so the different foot geometries see the same floor seed.
    for heel,pair in models.items():
        for f,floor in enumerate(FLOORS):
            for label,v,dist,n in [('3m',.2,3.,args.trials),('release',.2,None,args.trials),('touch',.15,3.,args.touch),('50m',.2,50.,args.long)]:
                offset={'3m':0,'release':10000,'touch':20000,'50m':30000}[label]
                for i in range(n):
                    seed=args.seed+offset+f+2*i
                    duration=(.32,1.3,2.72,4.88,6.16,8.22,4.52,6.30)[i%8] if dist is None else 360. if dist==50. else 32.
                    case=Case(seed,foot=FEET[heel],surface=floor,vx=v,seconds=duration,stop=3.5)
                    for profile,model in pair.items():
                        jobs.append((heel,profile,model,case,dist,label))
    groups=defaultdict(lambda:defaultdict(list))
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for h,f,label,p,row in pool.map(one,jobs,chunksize=1):
            groups['/'.join((h,f,label))][p].append(row)
    stats={}
    for k,models_ in groups.items():
        records={g:describe(rr) for g,rr in models_.items()}
        records['gain_pct']=100*(records['run']['speed_mps']/max(records['walk']['speed_mps'],.001)-1)
        stats[k]=records
        if k.endswith(('3m','50m')):
            print(k,round(records['walk']['speed_mps'],3),round(records['run']['speed_mps'],3),round(records['gain_pct'],1),
                  'stops',[(m['stops'],m['trials']) for m in (records['walk'],records['run'])], flush=True)
    write_json(args.out,dict(hashes=digests,seed=args.seed,groups=stats,episodes=groups))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--installed',action='store_true')
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--trials',type=int,default=8)
    p.add_argument('--touch',type=int,default=4)
    p.add_argument('--long',type=int,default=4)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--heels',nargs='+',choices=FEET,default=list(FEET))
    main(p.parse_args())
