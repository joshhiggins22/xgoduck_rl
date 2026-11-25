"""Summarize published heel models from their paired-course, perturbation and switch reports."""
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parent
FEET={'factory':'v2-stock','5mm':'v2-heel5','10mm':'v2-heel10'}
FLOORS=('hardwood','residential-carpet')
GAITS=('walk','run')


def load(path):
    return json.loads(path.read_text())


def report(models):
    stress=load(models/'stress.json')
    verify=load(models/'verify.json')
    pairs,nominal,perturbed,switched={},{'trials':0,'reached50':0,'falls':0,'stops':0},{'trials':0,'falls':0,'stops':0},{'trials':0,'falls':0,'stops':0}
    hashes={}
    worst_coast=worst_settle=worst_drift=0.
    for heel,foot in FEET.items():
        loc=models/heel
        audit=load(loc/'audit.json')
        switches=load(loc/'switches.json') if (loc/'switches.json').is_file() else None
        hashes[heel]={}
        for gait in GAITS:
            sha=hashlib.sha256((loc/(gait+'.onnx')).read_bytes()).hexdigest()
            assert sha==audit['hashes'][heel][gait]==stress['sha256'][foot+'/'+gait]==verify[heel][gait]['sha256']
            if switches: assert sha==switches['sha256'][gait]
            hashes[heel][gait]=sha
        for floor in FLOORS:
            k=heel+'/'+floor
            groups={g:audit['groups'][k+'/'+g] for g in ('3m','50m','touch','release')}
            paired={}
            for group,info in groups.items():
                for gait in GAITS:
                    entry=info[gait]
                    nominal['trials']+=entry['trials']
                    nominal['falls']+=entry['falls']
                    nominal['stops']+=entry['stops']
                    if group=='50m': nominal['reached50']+=entry['completed']
                    worst_coast=max(worst_coast,entry['worst_coast'] or 0.)
                    worst_settle=max(worst_settle,entry['worst_settle'] or 0.)
                    worst_drift=max(worst_drift,entry['worst_peak_drift'])
                paired[group]={'walk':info['walk'],'run':info['run'],'gain_pct':info['gain_pct']}
            pert={}
            for label in ['all','saved_calibration','no_saved_calibration']:
                rows=[r for r in stress['episodes'] if r['case']['foot']==foot and r['case']['surface']==floor
                      and (label=='all' or r['case']['calibrated']==(label=='saved_calibration'))]
                pert[label]={}
                for gait in GAITS:
                    a=[r['actors'][gait] for r in rows]
                    pert[label][gait]={'trials':len(a),'falls':sum(r['fell'] for r in a),
                          'stops':sum(r['stop_ok'] for r in a),'worst_coast':max((r['stop_peak_m'] or 0. for r in a),default=None)}
                    if label=='all':
                        for n in ('trials','falls','stops'): perturbed[n]+=pert[label][gait][n]
            pairs[k]={'course':paired,'hardware_variation':pert}
        if switches:
            for bucket in switches['episodes'].values():
                for pair in bucket:
                    s=pair['switch']
                    switched['trials']+=1
                    switched['falls']+=s['fell']
                    switched['stops']+=s['stop_ok']
    return dict(nominal=nominal,hardware_variation=perturbed,mode_switches=switched,hashes=hashes,
                worst_nominal_coast_m=worst_coast,worst_nominal_settle_s=worst_settle,worst_nominal_drift_m=worst_drift,
                pairs=pairs)


def markdown(data):
    rows=['# XGO-Duck v0.3.7 heel policies','',
          'Each heel size has a Walk and a Run policy trained across both floors. Select the heels physically fitted to the duck. The initial selection is 5 mm. Run uses the faster policy for those same heels. All performance figures below come from physics simulation.','',
          '| Heels | Floor | Walk, 3 m | Run, 3 m | Walk, 50 m | Run, 50 m | Speed gain over 50 m |',
          '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for k,group in data['pairs'].items():
        heel,floor=k.split('/')
        three,long=group['course']['3m'],group['course']['50m']
        rows.append(f"| {heel} | {floor} | {three['walk']['time_s']:.2f} s | {three['run']['time_s']:.2f} s | {long['walk']['speed_mps']:.3f} m/s | {long['run']['speed_mps']:.3f} m/s | +{long['gain_pct']:.1f}% |")
    n,s,t=data['nominal'],data['hardware_variation'],data['mode_switches']
    rows.extend(['','The benchmark commands full forward from a standing start and measures progress along the original heading. Walk and Run use the same start and floor realization. At each finish the stick returns to neutral. Other trials use partial stick or sudden releases at different points in the step.','',
        f"The nominal suite reached the 50 m finish {n['reached50']} times. Across {n['trials']} trials there were {n['falls']} falls and {n['stops']} sustained stops. The largest displacement from release was {data['worst_nominal_coast_m']*100:.1f} cm. The longest settling time on a successful stop was {data['worst_nominal_settle_s']:.2f} s. The largest sideways deviation was {data['worst_nominal_drift_m']:.2f} m.",'',
        f"With simulated motor strength, encoder, IMU and leg-delay differences, the policies stayed upright in {s['trials']-s['falls']} of {s['trials']} trials and completed {s['stops']} sustained stops. A stop requires at least two continuous seconds upright with planar motion below 2 cm/s and yaw below 8 degrees/s in every 0.2-second window.",''])
    if t['trials']:
        rows.extend([f"While changing Walk and Run at the control boundary, {t['trials']-t['falls']} of {t['trials']} trials stayed upright and {t['stops']} achieved a sustained stop.",''])
    rows.extend(['The residential-carpet preset models a level, compressible floor. It sinks 2.6–4.3 mm under a 4 N single-foot load across the simulated locations and heels. The stiffness and friction of the physical carpet have not been measured.','',
        'The accompanying results file includes individual heel/floor speeds, drift, tilt, stops by IMU calibration state and SHA-256 digests.'])
    return '\n'.join(rows)+'\n'


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,default=ROOT/'models')
    p.add_argument('--out',type=Path,default=ROOT)
    args=p.parse_args()
    data=report(args.models)
    (args.out/'results.json').write_text(json.dumps(data,indent=2)+'\n')
    (args.out/'RESULTS.md').write_text(markdown(data))
