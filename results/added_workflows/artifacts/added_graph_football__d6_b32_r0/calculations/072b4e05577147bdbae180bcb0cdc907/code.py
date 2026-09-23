
import numpy as np, itertools
from wwb_helpers import fit_surface
obs={o["mask"]:o["utility"] for o in observations}
masks=[format(i,'06b') for i in range(64)]
def xc(m): return 2*np.array([int(c) for c in m])-1
def des(cols):
    n=cols.shape[0]; D=[np.ones(n)]
    for i in range(6): D.append(cols[:,i])
    for i,j in itertools.combinations(range(6),2): D.append(cols[:,i]*cols[:,j])
    return np.column_stack(D)
om=list(obs.keys()); Di=np.array([des(xc(m)[None,:])[0] for m in om]); yi=np.array([obs[m] for m in om])
coef,*_=np.linalg.lstsq(Di,yi,rcond=None)
Dall=np.array([des(xc(m)[None,:])[0] for m in masks])
pred=dict(zip(masks,Dall@coef))
print("ODD-half degree2 predictions (popcount 1,3,5):")
for m in masks:
    if m.count('1')%2==1:
        print(m, m.count('1'), round(pred[m],4))
print()
print("Degree2 pred at single-feature masks vs known (from X,F obs):")
known={'100000':obs['100001'],'010000':obs['010001'],'001000':obs['001001'],'000100':obs['000101'],'000010':obs['000011'],'000001':0.5}
for m,v in known.items():
    print(m,"known~",round(v,4),"deg2 pred",round(pred[m],4))
# residuals sorted
res={m:pred[m]-obs[m] for m in om}
print("\nLargest residuals:")
for m,r in sorted(res.items(),key=lambda t:-abs(t[1]))[:12]: print(m, round(r,4), "pc",m.count('1'))
