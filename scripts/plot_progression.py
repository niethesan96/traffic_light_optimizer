"""Space/time green-state diagnostics; plot supports inspection, not causality claims."""
from pathlib import Path
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace')
    p.add_argument('--start',type=float,default=600)
    p.add_argument('--end',type=float,default=1200)
    p.add_argument('--block-length',type=float,default=300)
    p.add_argument('--speed',type=float,default=8.33)
    a=p.parse_args()
    rows=list(csv.DictReader(open(a.trace)))
    times=sorted({float(r['time']) for r in rows if a.start<=float(r['time'])<=a.end})
    if not times:
        p.error('No trace records in requested interval')
    lookup={(r['agent'],float(r['time'])):r for r in rows}
    fig,axes=plt.subplots(3,1,figsize=(12,6),layout='constrained')
    for row,ax in enumerate(axes,1):
        names=[f'agent_{row}_{col}' for col in range(1,4)]
        data=np.array([[int(lookup[(name,t)]['ew_green']) for t in times] for name in names])
        ax.imshow(data,aspect='auto',interpolation='nearest',origin='lower',
            extent=[times[0],times[-1],-0.5,2.5],cmap=ListedColormap(['#fee2e2','#15803d']),vmin=0,vmax=1)
        ax.set_yticks(range(3),names)
        ax.set_title(f'East–west corridor {row}: green availability by signal',fontsize=10)
        # Nominal travel-time guide, not fitted to policy behavior.
        travel=2*a.block_length/a.speed
        for start in np.arange(times[0],times[-1]-travel,90):
            ax.plot([start,start+travel],[0,2],color='#111827',alpha=.35,linewidth=.8)
        ax.set_xlabel(f'Simulation time (s); nominal guides: {a.block_length:g} m blocks at {a.speed:g} m/s')
    fig.suptitle('Signal progression diagnostic — verify with stops and throughput')
    fig.savefig(Path(a.trace).with_suffix('.png'),dpi=160)


if __name__=='__main__':
    main()
