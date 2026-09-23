
import numpy as np, itertools
masks=[format(i,'06b') for i in range(64)]
X=np.array([[1 if c=='1' else -1 for c in m] for m in masks],float)
y=np.full(64,np.nan)
for o in observations: y[masks.index(o['mask'])]=o['utility']
obs=np.where(np.isfinite(y))[0]; Xo=X[obs]; yo=y[obs]
idx={c:i for i,c in enumerate("ABCDEF")}
def mono(x,c):
    v=np.ones(x.shape[0])
    for i in c: v=v*x[:,i]
    return v
CDEbase=[(idx['C'],),(idx['D'],),(idx['E'],),(idx['C'],idx['D']),(idx['C'],idx['E']),(idx['D'],idx['E']),(idx['C'],idx['D'],idx['E'])]
def design_full(x):  # 1+7+3+21
    cols=[np.ones(x.shape[0])]
    cde=[mono(x,c) for c in CDEbase]
    cols+=cde
    A=mono(x,(idx['A'],)); B=mono(x,(idx['B'],)); F=mono(x,(idx['F'],))
    cols+=[A,B,F]
    for v in (A,B,F):
        cols+=[v*c for c in cde]
    return np.column_stack(cols)
M=design_full(Xo); p=M.shape[1]
# penalty: none on intercept/chde/mains (first 11), lambda on last 21
def loo_ridge(M,yy,lam,penmask):
    P=np.diag(penmask*lam); A=np.linalg.inv(M.T@M+P)
    b=A@(M.T@yy); h=np.einsum('ij,jk,ik->i',M,A,M)
    r=(yy-M@b)/np.clip(1-h,1e-9,None)
    return np.sqrt(np.mean(r**2)),b,A
penmask=np.zeros(p); penmask[11:]=1.0
best=None
for lam in np.logspace(-3,3,50):
    s,b,A=loo_ridge(M,yo,lam,penmask)
    if best is None or s<best[0]: best=(s,lam,b)
print("hierarchical ridge LOO:",round(best[0],4),"lam",round(best[1],3))
# unpenalized
s,b,A=loo_ridge(M,yo,1e-9,penmask); print("saturated (lam~0) LOO:",round(s,4))
# fair LOO of pooled CDE+ABF for reference
Mpool=np.column_stack([np.ones(len(obs))]+[mono(Xo,c) for c in CDEbase]+[mono(Xo,c) for c in [(idx['A'],),(idx['B'],),(idx['F'],)]])
print("pooled LOO:",round(loo_ridge(Mpool,yo,1e-9,np.zeros(Mpool.shape[1]))[0],4))
# also try penalty only on triple*ABF interactions? skip
# final predictions with best hierarchical
_,lam,b=best
ufull=design_full(X)  # 64
pred=np.clip(ufull@b,0,1)
unobs=np.setdiff1d(np.arange(64),obs)
print("\nhierarchical preds unobserved:")
for k in np.argsort(unobs): print("  ",masks[unobs[k]],round(float(pred[unobs[k]]),3))
