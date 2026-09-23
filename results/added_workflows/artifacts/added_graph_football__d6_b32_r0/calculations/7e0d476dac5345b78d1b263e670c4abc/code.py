
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
for name,deg in [("main",1),("pair",2)]:
    p=fit_surface(observations, degree=deg)
    print(name,"type",type(p))
    if isinstance(p,dict):
        ks=list(p.keys())[:3]; print(" keys sample",ks)
        P={m:float(p[m]) for m in masks if m in p}
    else:
        arr=np.asarray(p).ravel(); print(" len",arr.size)
        P={m:float(v) for m,v in zip(masks,arr)} if arr.size==64 else None
    if P:
        odd=[m for m in masks if m.count('1')%2==1]
        err=[abs(P[m]-tbl[m]) for m in odd]
        print("  MAE vs tbl(odd)",round(np.mean(err),4),"max",round(np.max(err),4))
        for m in ["000010","000001","111101","001110","111110"]:
            print("  ",m,round(tbl[m],4),round(P[m],4))
