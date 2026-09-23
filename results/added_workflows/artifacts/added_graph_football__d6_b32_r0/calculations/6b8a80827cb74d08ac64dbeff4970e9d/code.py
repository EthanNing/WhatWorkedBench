
import numpy as np, json
obs={o["mask"]:o["utility"] for o in observations}
masks=[format(i,'06b') for i in range(64)]
def nbrs(m):
    b=[int(c) for c in m]; out=[]
    for i in range(6):
        b2=b.copy(); b2[i]^=1; out.append(''.join(map(str,b2)))
    return out
single={'100000':0.8046797541146143,'010000':0.8247075153678366,'001000':0.8134047194130478,
        '000100':0.8135369158569634,'000010':0.6422103245422696,'000001':0.5}
final={}
for m in masks:
    if m in obs: final[m]=obs[m]
    elif m in single: final[m]=single[m]
    else:
        vals=[obs[k] for k in nbrs(m)]
        assert all(k in obs for k in nbrs(m))
        final[m]=float(np.median(vals))
# sanity: all finite, in [0,1]
assert all(np.isfinite(v) and 0<=v<=1 for v in final.values())
# edge difference stats
diffs=[]
for m in masks:
    for k in nbrs(m):
        if m<k: diffs.append(final[k]-final[m])
diffs=np.array(diffs); print("n masks",len(final),"n edges",len(diffs))
print("edge diff mean|.|",round(np.mean(np.abs(diffs)),4),"max|.|",round(np.max(np.abs(diffs)),4))
print(json.dumps({m:round(final[m],6) for m in masks}))
