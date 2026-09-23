
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
def dm(x,ex): return np.column_stack([np.ones(x.shape[0])]+[mono(x,c) for c in CDE+ex])
models={
 'm0 CDE+A+B+F+BF':[(iA,),(iB,),(iF,),(iB,iF)],
 'm1 +AC':[(iA,),(iB,),(iF,),(iB,iF),(iA,iC)],
 'm2 +AC+AF+AB':[(iA,),(iB,),(iF,),(iB,iF),(iA,iC),(iA,iF),(iA,iB)],
 'm3 +AC+AD+AE':[(iA,),(iB,),(iF,),(iB,iF),(iA,iC),(iA,iD),(iA,iE)],
 'm4 +AC+ACD+ACE+ACDE':[(iA,),(iB,),(iF,),(iB,iF),(iA,iC),(iA,iC,iD),(iA,iC,iE),(iA,iC,iD,iE)],
}
def loo(M,yy,lam=1e-8):
    Am=np.linalg.inv(M.T@M+lam*np.eye(M.shape[1])); b=Am@(M.T@yy)
    h=np.einsum('ij,jk,ik->i',M,Am,M); r=(yy-M@b)/np.clip(1-h,1e-6,None)
    return np.sqrt(np.mean(r**2)),b,np.max(np.abs(yy-M@b))
unobs=np.setdiff1d(np.arange(64),obs)
res={}
for k,ex in models.items():
    M=dm(Xo,ex); s,b,mx=loo(M,yo); res[k]=(b,ex); print(f"{k:22s} p={M.shape[1]:2d} LOO {s:.4f} maxres {mx:.3f}")
cols=list(models)
pred={k:np.clip(dm(X,res[k][1])@res[k][0],0,1) for k in cols}
print("\nunobs  "+" ".join(f"{c[:14]:>15s}" for c in cols))
for k in np.argsort(unobs):
    print(masks[unobs[k]]+" "+" ".join(f"{pred[c][unobs[k]]:15.3f}" for c in cols))
# partner(A-irrelevant) predictions
part=np.array([y[masks.index(masks[u][0]+('0' if masks[u][0]=='1' else '1')+masks[u][1:])] if (masks.index(masks[u][0]+('0' if masks[u][0]=='1' else '1')+masks[u][1:])) in obs else np.nan for u in unobs])
print("\npartner vals:", [ (masks[unobs[j]], None if np.isnan(part[j]) else round(float(part[j]),3)) for j in range(len(unobs))])
