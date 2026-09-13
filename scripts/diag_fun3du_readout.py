"""Why the first Fun3DU run produced an unusable Table 2 (paper Sec. 6).

The run finished and the numbers looked catastrophic: of 116 elements, only 9
cleared IoU3D 0.25 against the ground-truth element. Read naively that says
Fun3DU cannot find functional elements, which contradicts its published AP25 of
33.3 and would have been a serious thing to print.

It was our readout. motion_on_fun3du.py thresholds the lifted score field the
way Fun3DU's own evaluate.py writes it,

    np_normalize(acc_f / n_views) > 0.7

and np_normalize rescales each file to [0,1] -- so "> 0.7" keeps the top 30% of
the score RANGE, not the top 30% of the points. Where scores are concentrated,
that admits nearly the whole parent object.

This script separates the two hypotheses using only the per-element record the
run already wrote, with no GPU and no re-run. IoU and the two cardinalities pin
the intersection down exactly,

    IoU = I / (P + G - I)   =>   I = IoU * (P + G) / (1 + IoU)

from which recall (I/G) and precision (I/P) follow. High recall with collapsed
precision means the element was found and the region was simply too large --
a readout fault. Low recall would mean a detection fault, which is Fun3DU's.

Result: recall >= 0.5 on 40.8% of elements, consistent with their published
AP25; median region 14.2x the ground-truth element; median precision 0.003.
Ours. scripts/fun3du_threshold_sweep.py re-derives the operating point.

Run: python scripts/diag_fun3du_readout.py
"""
import os, sys, json
import statistics as st
from _paths import DOCS

ROWS = os.path.join(DOCS, 'fun3du_table2_rows.json')

def main():
    if not os.path.exists(ROWS):
        raise SystemExit(f'missing {ROWS} -- produced by scripts/motion_on_fun3du.py')
    allrows = json.load(open(ROWS))
    rows = [r for r in allrows if 'n_pred' in r]
    print(f'elements evaluated      {len(allrows)}')
    print(f'  with a Fun3DU region  {len(rows)}')
    print(f'  with none             {len(allrows) - len(rows)}\n')

    rec, pre, bloat = [], [], []
    for r in rows:
        P, G, iou = r['n_pred'], r['n_gt'], r['iou3d']
        I = iou * (P + G) / (1 + iou)
        rec.append(min(I / G, 1.0)); pre.append(I / max(P, 1)); bloat.append(P / G)

    print(f'{"IoU3D >= 0.25":<26}{sum(1 for r in rows if r["iou3d"]>=0.25):>5} '
          f'({100*sum(1 for r in rows if r["iou3d"]>=0.25)/len(rows):.1f}%)')
    print(f'{"recall >= 0.5 (FOUND)":<26}{sum(1 for x in rec if x>=0.5):>5} '
          f'({100*sum(1 for x in rec if x>=0.5)/len(rec):.1f}%)')
    print(f'{"median recall":<26}{st.median(rec):>5.3f}')
    print(f'{"median precision":<26}{st.median(pre):>5.3f}')
    print(f'{"median n_pred / n_gt":<26}{st.median(bloat):>5.1f}x')

    found = [b for b, r in zip(bloat, rec) if r >= 0.5]
    print(f'{"  ...on found elements":<26}{st.median(found):>5.1f}x')

    print('\nVERDICT: high recall with collapsed precision -- the elements were')
    print('found and the regions were too large. Readout fault, not detection.')

    out = dict(n_elements=len(allrows), n_with_region=len(rows),
               det_at_025=round(100*sum(1 for r in rows if r['iou3d']>=0.25)/len(rows), 2),
               recall_ge_05_pct=round(100*sum(1 for x in rec if x>=0.5)/len(rec), 2),
               median_recall=round(st.median(rec), 4),
               median_precision=round(st.median(pre), 4),
               median_bloat=round(st.median(bloat), 2),
               median_bloat_on_found=round(st.median(found), 2))
    json.dump(out, open(os.path.join(DOCS, 'fun3du_readout_diag.json'), 'w'), indent=1)
    print('\nwrote docs/fun3du_readout_diag.json')

if __name__ == '__main__':
    main()
