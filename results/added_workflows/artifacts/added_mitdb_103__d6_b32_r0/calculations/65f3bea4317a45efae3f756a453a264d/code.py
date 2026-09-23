
import numpy as np, itertools
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
 'CDE+BF':[(iB,),(iF,)],
 'CDE+BF+BFi':[(iB,),(iF,),(iB,iF)],
 'CDE+A+BF+BFi':[(iA,),(iB,),(iF,),(iB,iF)],
 'CDE+B+F+ABi+BFi':[(iB,),(iF,),(iA,iB),(iB,iF)],
}
def loo(M,yy,lam=1e-9):
    Am=np.linalg.inv(M.T@M+lam*np.eye(M.shape[1])); b=Am@(M.T@yy)
    h=np.einsum('ij,jk,ik->i',M,Am,M); r=(yy-M@b)/np.clip(1-h,1e-6,None)
    return np.sqrt(np.mean(r**2)),b
fitb={}
for k,ex in mods.items():
    M=dm(Xo,ex); s,b=loo(M,yo); fitb[k]=(s,b,ex); print(f"{k:20s} p={M.shape[1]:2d} LOO {s:.4f}")
unobs=np.setdiff1d(np.arange(64),obs)
cols=list(mods)
preds={k:np.clip(dm(X,fitb[k][2])@fitb[k][1],0,1) for k in cols}
print("\nunobs  "+" ".join(f"{k:>16s}" for k in cols))
for k in np.argsort(unobs):
    print(masks[unobs[k]]+" "+" ".join(f"{preds[c][unobs[k]]:16.3f}" for c in cols))
