
import numpy as np, itertools, collections, random
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
singles=['100000','010000','001000','000100','000010']
fixed=['000000','111111']+singles
cand=[m for m in allm if popAE(m)>=2 and m!='111111']
edges=[(m,m[:i]+'1'+m[i+1:]) for m in allm for i in range(6) if m[i]=='0']
def zeroset(obs):
    z=set(obs)
    for s in singles:
        if s in obs: z.add(s[:5]+str(1-int(s[5])))
    if '000000' in obs: z.add('000001')
    return z
def crit(sel,B=B2,ridge=1e-6):
    X=np.array([B[idx[m]] for m in sel]); p=X.shape[1]
    Minv=np.linalg.inv(X.T@X+ridge*np.eye(p))
    zero=zeroset(sel); tot=0.0; ex=0
    for u,v in edges:
        U=[z for z in (u,v) if z not in zero]
        if not U: ex+=1; continue
        for a in U:
            for b in U: tot+=B[idx[a]]@Minv@B[idx[b]]
    return tot,ex
nfix=len(fixed)
def search(sel,passes=5):
    sel=list(sel); cur=crit(sel)[0]
    for _ in range(passes):
        imp=False
        for i in range(nfix,len(sel)):
            for c in cand:
                if c in sel: continue
                t=sel[:]; t[i]=c; e=crit(t)[0]
                if e<cur-1e-7: sel=t; cur=e; imp=True; break
            if imp: break
        if not imp: break
    return sel,cur
random.seed(1); best=None
for k in range(10):
    st=fixed+list(np.random.choice(cand,27,replace=False))
    s,e=search(st)
    if best is None or e<best[1]: best=(s,e); print("restart",k,"-> %.2f"%e,"exact",crit(s)[1])
sel,e=best
print("BEST deg2 edgeMSE %.2f exact %d deg1 %.2f"%(e,crit(sel)[1],crit(sel,B1)[0]))
print("PURCHASES:",sorted([m for m in sel if m not in ('000000','111111')]))
print("hist",dict(collections.Counter(popAE(m) for m in sel)))
