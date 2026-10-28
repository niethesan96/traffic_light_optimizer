"""Estimate nominal downstream green coverage from recorded signal timelines."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np


def analyze(path,block_length=300,speed=8.33):
    rows=list(csv.DictReader(open(path)))
    times=sorted({float(r['time']) for r in rows})
    if len(times)<2:
        raise ValueError('Need multiple signal timestamps')
    dt=float(np.median(np.diff(times)))
    lag=max(1,round(block_length/speed/dt))
    names=sorted({r['agent'] for r in rows})
    lookup={(r['agent'],float(r['time'])):r for r in rows}
    series={(n,key):np.array([int(lookup[(n,t)][key]) for t in times])
            for n in names for key in ('ew_green','ns_green')}
    links=[]
    total_green=total_covered=0
    for origin in names:
        r,c=map(int,origin.split('_')[-2:])
        for rr,cc,key in ((r,c-1,'ew_green'),(r,c+1,'ew_green'),
                          (r-1,c,'ns_green'),(r+1,c,'ns_green')):
            destination=f'agent_{rr}_{cc}'
            if destination not in names:
                continue
            departed=series[(origin,key)][:-lag].astype(bool)
            arrivals=series[(destination,key)][lag:].astype(bool)
            green=int(departed.sum())
            covered=int((departed&arrivals).sum())
            total_green+=green
            total_covered+=covered
            links.append(dict(origin=origin,destination=destination,
                coverage=covered/green if green else None))
    return dict(trace=str(path),resolution_seconds=dt,nominal_travel_seconds=block_length/speed,
        sampled_lag_seconds=lag*dt,
        resolution_adequate=dt<=2,
        nominal_green_transfer_probability=total_covered/total_green if total_green and dt<=2 else None,
        links=links if dt<=2 else [],
        interpretation=('Signal alignment diagnostic, not vehicle-weighted green arrivals or causal proof'
            if dt<=2 else 'Insufficient temporal resolution: rerun evaluation with physics-step signal traces'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace')
    p.add_argument('--block-length',type=float,default=300)
    p.add_argument('--speed',type=float,default=8.33)
    a=p.parse_args()
    result=analyze(a.trace,a.block_length,a.speed)
    Path(a.trace).with_suffix('.progression.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='links'}))


if __name__=='__main__':
    main()
