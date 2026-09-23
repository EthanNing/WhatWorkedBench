
import numpy as np, itertools
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
pred={m:des(xc(m)[None,:])[0]@coef for m in masks}
def nbrs(m):
    b=np.array([int(c) for c in m])
    out=[]
    for i in range(6):
        b2=b.copy(); b2[i]^=1; out.append(''.join(map(str,b2)))
    return out
odd=[m for m in masks if m.count('1')%2==1]
rows=[]
for m in odd:
    na=np.mean([obs[k] for k in nbrs(m)])
    rows.append((m,m.count('1'),pred[m],na,pred[m]-na))
rows.sort(key=lambda t:-abs(t[4]))
print("mask pc deg2 nbrAvg diff(deg2-nbr)")
for r in rows: print(r[0], r[1], round(r[2],4), round(r[3],4), round(r[4],4))
