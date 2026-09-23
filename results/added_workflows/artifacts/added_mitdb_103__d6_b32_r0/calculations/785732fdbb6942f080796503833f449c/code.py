
import numpy as np
masks=[format(i,'06b') for i in range(64)]
X=np.array([[1 if c=='1' else -1 for c in m] for m in masks],float)
y=np.full(64,np.nan)
for o in observations: y[masks.index(o['mask'])]=o['utility']
obs=np.where(np.isfinite(y))[0]; Xo=X[obs]; yo=y[obs]
iA,iB,iC,iD,iE,iF=0,1,2,3,4,5
def mono(x,c):
    v=np.ones(x.shape[0])
    for i in c: v=v*x[:,i]
    return v
CDE=[(iC,),(iD,),(iE,),(iC,iD),(iC,iE),(iD,iE),(iC,iD,iE)]
def dm(x,extras): return np.column_stack([np.ones(x.shape[0])]+[mono(x,c) for c in CDE+extras])
mods={
 'CDE+BF':dm(Xo,[(iB,),(iF,)]),
 'CDE+BF+BFint':dm(Xo,[(iB,),(iF,),(iB,iF)]),
 'CDE+A+BF+BFint':dm(Xo,[(iA,),(iB,),(iF,),(iB,iF)]),
 'CDE+A+BF+BFint+AB+AF':dm(Xo,[(iA,),(iB,),(iF,),(iB,iF),(iA,iB),(iA,iF)]),
 'CDE+ABF+BFint':dm(Xo,[(iA,),(iB,),(iF),(iB,iF)]),
}
def loo(M,yy,lam=1e-9):
    P=lam*np.eye(M.shape[1]); 
    Am=np.linalg.inv(M.T@M+P); b=Am@(M.T@yy)
    h=np.einsum('ij,jk,ik->i',M,Am,M); r=(yy-M@b)/np.clip(1-h,1e-6,None)
    return np.sqrt(np.mean(r**2)),b
for k,M in mods.items():
    s,b=loo(M,yo); print(f"{k:24s} p={M.shape[1]:2d} LOO {s:.4f}")

# Does A matter? t-test style: compare residuals of CDE+BF+BFint vs +A
# Also: fit full 6-var degree2 to get A-related terms with new data
import itertools
sp1=[(i,) for i in range(6)]; sp2=sp1+list(itertools.combinations(range(6),2))
M=dm(Xo,[(i,) for i in range(6)]+list(itertools.combinations(range(6),2)))
s,b=loo(M,yo); print("full degree2 LOO",round(s,4))
