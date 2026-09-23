
import numpy as np, itertools, collections
allm=[''.join(map(str,b)) for b in itertools.product([0,1],repeat=6)]
idx={m:i for i,m in enumerate(allm)}
def bv(m): return np.array([int(c) for c in m],float)
def bas2(x):
    t=[1.0]; t+=list(x)
    for i in range(6):
        for j in range(i+1,6): t.append(x[i]*x[j])
    return np.array(t)
B2=np.array([bas2(bv(m)) for m in allm])
B1=np.array([[1.0]+list(bv(m)) for m in allm])
def popAE(m): return sum(int(c) for c in m[:5])
# zero-error set = observed(design) U deduced: low cells with popAE<=1 whose base single is measured
def zeroset(obs):
    z=set(obs)
    for s in ['100000','010000','001000','000100','000010']:
        if s in obs:
            z.add(s[:5]+str(1-int(s[5])))  # flip F in low position
    if '000000' in obs: z.add('000001')
    return z
singles=['100000','010000','001000','000100','000010']
fixed=['000000','111111']+singles
cand=[m for m in allm if popAE(m)>=2 and m!='111111']
edges=[(m,m[:i]+'1'+m[i+1:]) for m in allm for i in range(6) if m[i]=='0']
def edge_mse(obs_set,B=B2,ridge=1e-6):
    X=np.array([B[idx[m]] for m in obs_set]); p=X.shape[1]
    Minv=np.linalg.inv(X.T@X+ridge*np.eye(p))
    zero=zeroset(obs_set); tot=0.0; exact=0
    for u,v in edges:
        U=[z for z in (u,v) if z not in zero]
        if not U: exact+=1; continue
        for a in U:
            for b in U: tot+=B[idx[a]]@Minv@B[idx[b]]
    return tot, exact
print("fixed design:",edge_mse(fixed))
def search(sel, passes=8):
    sel=list(sel); cur=edge_mse(sel)[0]; nfix=len(fixed)
    for _ in range(passes):
        improved=False
        for i in range(nfix,len(sel)):
            for c in cand:
                if c in sel: continue
                trial=sel[:]; trial[i]=c
                e=edge_mse(trial)[0]
                if e<cur-1e-7:
                    sel=trial; cur=e; improved=True; break
            if improved: break
        if not improved: break
    return sel,cur
import random
best=None
random.seed(0)
starts=[fixed+list(np.random.choice(cand,27,replace=False)) for _ in range(6)]
for st in starts:
    s,e=search(st)
    if best is None or e<best[1]: best=(s,e)
sel,e=best
print("refined edgeMSE %.3f"%e,"exact",edge_mse(sel)[1])
print("design (34):",sorted(sel))
print("purchases (32):",sorted([m for m in sel if m!='000000' and m!='111111']))
print("hist",collections.Counter(popAE(m) for m in sel),"Fon",sum(m[5]=='1' for m in sel))
print("deg1 crit %.3f"%edge_mse(sel,B1)[0])
# count exact edges explicitly types
zero=zeroset(sel)
print("exact edge list count by type:",collections.Counter(
   ('both_obs' if (u in sel and v in sel) else 'obs+deduced') for u,v in edges if u in zero and v in zero))
