
import numpy as np, itertools
from wwb_helpers import fit_surface

masks=[format(i,'06b') for i in range(64)]
X=np.array([[1 if c=='1' else -1 for c in m] for m in masks],float)
y=np.full(64,np.nan)
for o in observations: y[masks.index(o['mask'])]=o['utility']
obs=np.where(np.isfinite(y))[0]

def terms(x,maxdeg):
    n=x.shape[0]; cols=[np.ones(n)]; names=['1']
    for d in range(1,maxdeg+1):
        for combo in itertools.combinations(range(6),d):
            v=np.ones(n)
            for i in combo: v=v*x[:,i]
            cols.append(v); names.append("".join("ABCDEF"[i] for i in combo))
    return np.column_stack(cols),names

def ridge_loo(M,yy,lams):
    n,p=M.shape; best=None
    for lam in lams:
        P=np.eye(p)*lam; P[0,0]=0
        A=np.linalg.inv(M.T@M+P)
        beta=A@(M.T@yy)
        h=np.einsum('ij,jk,ik->i',M,A,M)
        r=(yy-M@beta)/np.clip(1-h,1e-6,None)
        s=np.sqrt(np.mean(r**2))
        if best is None or s<best[0]: best=(s,lam,beta,A)
    return best

lams=np.logspace(-4,2,40)
print("=== LOO within even half ===")
res={}
for deg in (1,2,3):
    M,names=terms(X[obs],deg)
    s,lam,beta,A=ridge_loo(M,y[obs],lams)
    res[deg]=(s,lam,beta,A,names)
    print(f"deg{deg}: LOO rmse {s:.4f} (lam={lam:.3g}, p={M.shape[1]})")

# clamped predictions for deg3
for deg in (1,2,3):
    M,names=terms(X[obs],deg); s,lam,beta,A=res[deg]
    pred_loo=np.clip(M@beta,0,1)  # crude
print("deg3 largest |LOO resid| (unclamped):")
M,names=terms(X[obs],3); s,lam,beta,A=res[3]
h=np.einsum('ij,jk,ik->i',M,A,M); r=(y[obs]-M@beta)/np.clip(1-h,1e-6,None)
bad=np.argsort(-np.abs(r))[:10]
for k in bad: print("  ",masks[obs[k]], round(y[obs[k]],3), round((M@beta)[k],3), "loo",round(r[k],3))

# GP hamming kernel
def gp_loo(theta,noise):
    D=(X[obs][:,None,:]!=X[obs][None,:,:]).sum(-1)/6.0
    K=np.exp(-theta*D)+noise*np.eye(len(obs))
    Ki=np.linalg.inv(K)
    alpha=Ki@y[obs]
    loo=y[obs]-alpha/np.diag(Ki)
    return np.sqrt(np.mean(loo**2))
best=None
for th in np.linspace(0.2,6,30):
    for nz in [1e-6,1e-4,1e-3,1e-2]:
        s=gp_loo(th,nz)
        if best is None or s<best[0]: best=(s,th,nz)
print("GP hamming LOO rmse:",round(best[0],4),"theta",round(best[1],2),"noise",best[2])

# predict odd cells with deg3 ridge and gp
odd=np.setdiff1d(np.arange(64),obs)
Mfull,names=terms(X[odd],3); Min,names=terms(X[obs],3)
_,_,beta,_,_=res[3]
p3=Mfull@beta
print("\ndeg3 ridge predictions on odd half (clamped):")
for k in range(len(odd)): print("  ",masks[odd[k]], round(float(np.clip(p3[k],0,1)),3))
