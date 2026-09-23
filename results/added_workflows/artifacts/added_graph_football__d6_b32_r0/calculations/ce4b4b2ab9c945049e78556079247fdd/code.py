
import json, numpy as np
from wwb_helpers import fit_surface
obs={o["mask"]:o["utility"] for o in observations}
masks=[format(i,'06b') for i in range(64)]
def nbrs(m):
    b=[int(c) for c in m]; out=[]
    for i in range(6):
        b2=b.copy(); b2[i]^=1; out.append(''.join(map(str,b2)))
    return out
single={'100000':0.8046797541146143,'010000':0.8247075153678366,'001000':0.8134047194130478,
        '000100':0.8135369158569634,'000010':0.6422103245422696,'000001':0.5}
tbl={}
for m in masks:
    if m in obs: tbl[m]=obs[m]
    elif m in single: tbl[m]=single[m]
    else: tbl[m]=float(np.median([obs[k] for k in nbrs(m)]))
for name in ["ridge_main","ridge_pair"]:
    pass
p1=fit_surface(observations, degree=1)
p2=fit_surface(observations, degree=2)
def norm(p):
    if isinstance(p,dict): return p
    return {m:float(v) for m,v in zip(masks,p)}
P1,P2=norm(p1),norm(p2)
odd=[m for m in masks if m.count('1')%2==1]
for nm,P in [("deg1",P1),("deg2",P2)]:
    err=[abs(P[m]-tbl[m]) for m in odd]
    print(nm,"MAE vs my table (odd):",round(np.mean(err),4),"max",round(np.max(err),4))
print("sample odd: mask  mytable  deg1   deg2")
for m in ["000010","000001","111101","001110","111110","111010"]:
    print(m, round(tbl[m],4), round(P1[m],4), round(P2[m],4))
