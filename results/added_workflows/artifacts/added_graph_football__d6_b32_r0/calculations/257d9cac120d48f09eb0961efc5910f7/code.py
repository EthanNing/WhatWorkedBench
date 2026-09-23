
import numpy as np
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
edges=[(m,k) for m in masks for k in nbrs(m) if m<k]
def ediff(P): return np.array([P[k]-P[m] for m,k in edges])
d_tbl=ediff(tbl)
for name,deg in [("deg1",1),("deg2",2)]:
    P={m:float(v) for m,v in fit_surface(observations,degree=deg)['predictions'].items()}
    dP=ediff(P)
    print(name,"edge-diff MAE vs my table:",round(np.mean(np.abs(dP-d_tbl)),4),
          "corr:",round(np.corrcoef(dP,d_tbl)[0,1],4))
print("my table mean|ediff|",round(np.mean(np.abs(d_tbl)),4))
