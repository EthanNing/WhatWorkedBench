
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
sp1=[(i,) for i in range(6)]
sp2=sp1+list(itertools.combinations(range(6),2))
def dm(x,extras): return np.column_stack([np.ones(x.shape[0])]+[mono(x,c) for c in extras])
models={
 'CDE+A+B+F+BF':CDE+[(iA,),(iB,),(iF,),(iB,iF)],
 'CDE+A+B+F+BF+AC+AF+AB':CDE+[(iA,),(iB,),(iF,),(iB,iF),(iA,iC),(iA,iF),(iA,iB)],
 'd2+CDE':sp2+[(iC,iD,iE)],
 'd2+CDE+ABFi':sp2+[(iC,iD,iE),(iB,iF)],
 'd2+CDE+ABFi+ACi+AFi':sp2+[(iC,iD,iE),(iB,iF),(iA,iC),(iA,iF)],
}
def loo(M,yy,lam=1e-8):
    Am=np.linalg.inv(M.T@M+lam*np.eye(M.shape[1])); b=Am@(M.T@yy)
    h=np.einsum('ij,jk,ik->i',M,Am,M); r=(yy-M@b)/np.clip(1-h,1e-6,None)
    return np.sqrt(np.mean(r**2)),b
unobs=np.setdiff1d(np.arange(64),obs)
out={}
for k,ex in models.items():
    M=dm(Xo,ex); s,b=loo(M,yo); out[k]=(s,b,ex)
    print(f"{k:28s} p={M.shape[1]:2d} LOO {s:.4f}")
cols=list(models)
preds={k:np.clip(dm(X,out[k][2])@out[k][1],0,1) for k in cols}
print("\nunobs  "+" ".join(f"{c[:16]:>17s}" for c in cols))
for k in np.argsort(unobs):
    print(masks[unobs[k]]+" "+" ".join(f"{preds[c][unobs[k]]:17.3f}" for c in cols))
