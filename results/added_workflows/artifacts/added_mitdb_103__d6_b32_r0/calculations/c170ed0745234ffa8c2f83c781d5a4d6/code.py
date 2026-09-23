
import numpy as np
from wwb_helpers import fit_surface, suggest_experiments

masks=[format(i,'06b') for i in range(64)]
X=np.array([[1 if c=='1' else -1 for c in m] for m in masks],float)
y=np.full(64,np.nan)
for o in observations:
    y[masks.index(o['mask'])]=o['utility']
obs_idx=np.where(np.isfinite(y))[0]
print("observed:",len(obs_idx))

def MM(x,deg=2):
    n=x.shape[0]; t=[np.ones(n)]
    for i in range(6): t.append(x[:,i])
    if deg>=2:
        for i in range(6):
            for j in range(i+1,6): t.append(x[:,i]*x[:,j])
    return np.column_stack(t)

# main effects (clean, orthogonal)
print("\nMAIN EFFECTS (mean y | +1) - (mean y | -1):")
for i in range(6):
    p=y[(X[:,i]==1)&np.isfinite(y)].mean(); q=y[(X[:,i]==-1)&np.isfinite(y)].mean()
    print(" ABCDEF"[i+1] if False else "ABCDEF"[i], round(p,4), round(q,4), "eff",round(p-q,4))

# OLS degree1 and degree2 on even half
for deg in (1,2):
    M=MM(X[obs_idx],deg)
    beta,res,rank,sv=np.linalg.lstsq(M,y[obs_idx],rcond=None)
    pred=M@beta
    print(f"\ndeg{deg}: rank {rank} rmse {np.sqrt(np.mean((pred-y[obs_idx])**2)):.4f}")
    names=['1']+list("ABCDEF")
    if deg>=2:
        for i in range(6):
            for j in range(i+1,6): names.append("ABCDEF"[i]+"ABCDEF"[j])
    for n,b in zip(names,beta): print(f"  {n:5s} {b: .4f}")

# LOO for deg2
M=MM(X[obs_idx],2)
lev=np.einsum('ij,jk,ik->i',M,np.linalg.inv(M.T@M),M)
pred_loo=(M@np.linalg.lstsq(M,y[obs_idx],rcond=None)[0]-lev*y[obs_idx])/(1-lev)
print("\nmax |LOO resid| deg2:",np.max(np.abs(pred_loo-y[obs_idx])))
bad=np.argsort(-np.abs(pred_loo-y[obs_idx]))[:8]
print([ (masks[obs_idx[k]], round(y[obs_idx[k]],3), round(pred_loo[k],3)) for k in bad])
