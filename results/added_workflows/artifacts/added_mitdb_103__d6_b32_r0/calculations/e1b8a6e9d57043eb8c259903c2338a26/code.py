
import numpy as np, itertools
from wwb_helpers import suggest_experiments, fit_surface
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
def cde_cols(x):
    C,D,E=[x[:,idx[k]] for k in "CDE"]
    return [(C**c)*(D**d)*(E**e) for c,d,e in [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]]
def Dbar(x,extras): return np.column_stack([np.ones(x.shape[0])]+cde_cols(x)+[mono(x,c) for c in extras])
spec2=[(i,) for i in range(6)]+list(itertools.combinations(range(6),2))
def build(x,sp): return np.column_stack([np.ones(x.shape[0])]+[mono(x,c) for c in sp])

def fitpred(tr,te,Mfn):
    Mtr=Mfn(X[tr]); b=np.linalg.lstsq(Mtr,y[tr],rcond=None)[0]
    return Mfn(X[te])@b

unobs=np.setdiff1d(np.arange(64),obs)
models={
 'CDE+ABF':lambda x:Dbar(x,[(idx['A'],),(idx['B'],),(idx['F'],)]),
 'd2':lambda x:build(x,spec2),
 'd2+CDE':lambda x:build(x,spec2+[(idx['C'],idx['D'],idx['E'])]),
}
P=np.column_stack([fitpred(obs,unobs,m) for m in models.values()])
P=np.clip(P,0,1)
spread=P.max(1)-P.min(1)
print("suggest_experiments:")
for n in (3,6):
    try: print(" count",n,"->",suggest_experiments(observations,count=n,degree=2))
    except Exception as e: print("  err",e)
print("\ntop-uncertainty unobserved cells (ensemble spread):")
order=np.argsort(-spread)[:10]
for k in order:
    print("  ",masks[unobs[k]],"preds",np.round(P[k],3),"spread",round(float(spread[k]),3))
# show CDE+ABF predictions for all unobserved
m0=fitpred(obs,unobs,models['CDE+ABF']); m0=np.clip(m0,0,1)
print("\nCDE+ABF preds for all unobserved:")
for k in np.argsort(unobs): print("  ",masks[unobs[k]], round(float(m0[k]),3))
