"""Render exportable three-seed stage comparisons from compact CSV tables."""
import csv
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def render(folder):
    with (folder/'paired_stage_deltas.csv').open() as f: ap=list(csv.DictReader(f))
    with (folder/'paired_coverage_deltas.csv').open() as f: cov=list(csv.DictReader(f))
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    colors=['#1b9e77','#d95f02','#7570b3']
    for ax,metric in zip(axes[0],('AP','APs')):
        stages=['D1','D2','D3']; vals=[]
        for seed,color in enumerate(colors):
            y=[float(next(r[f'delta_{metric}_pp'] for r in ap if int(r['seed'])==seed and r['stage']==s)) for s in stages]
            vals.append(y); ax.plot(stages,y,'o-',color=color,label=f'Seed {seed}',linewidth=2)
        ax.plot(stages,[statistics.mean(v[i] for v in vals) for i in range(3)],'s--',color='#222222',label='Mean',linewidth=2)
        ax.set_title(f'Formal COCO {metric}: C2 minus C0'); ax.set_ylabel('Difference (percentage points)')
    for ax,threshold in zip(axes[1],(.5,.75)):
        stages=['E','D1','D2','D3']; vals=[]
        for seed,color in enumerate(colors):
            y=[float(next(r['correct_delta_pp'] for r in cov if int(r['seed'])==seed and r['stage']==s and r['group']=='all' and float(r['iou_threshold'])==threshold)) for s in stages]
            vals.append(y); ax.plot(stages,y,'o-',color=color,label=f'Seed {seed}',linewidth=2)
        ax.plot(stages,[statistics.mean(v[i] for v in vals) for i in range(4)],'s--',color='#222222',label='Mean',linewidth=2)
        ax.set_title(f'Class-correct coverage, IoU {threshold:g}: C2 minus C0'); ax.set_ylabel('Difference (percentage points)')
    for ax in axes.flat:
        ax.axhline(0,color='#888888',linewidth=.8); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('P2-SFI across three paired seeds | 548 images | FP32 | no retraining',fontsize=13)
    fig.savefig(folder/'stage_comparison.png',dpi=180)
    fig.savefig(folder/'stage_comparison.svg')
    plt.close(fig)

if __name__=='__main__': render(Path(__file__).resolve().parent)
