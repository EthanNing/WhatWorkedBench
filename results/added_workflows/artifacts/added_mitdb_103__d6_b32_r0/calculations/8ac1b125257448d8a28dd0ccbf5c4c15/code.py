
import numpy as np, itertools
masks=[format(i,'06b') for i in range(64)]
X=np.array([[1 if c=='1' else -1 for c in m] for m in masks],float)
y=np.full(64,np.nan)
for o in observations: y[masks.index(o['mask'])]=o['utility']
obs=np.where(np.isfinite(y))[0]
Xo=X[obs]; yo=y[obs]
idx={c:i for i,c in enumerate("ABCDEF")}

def mono(x,combo):
    v=np.ones(x.shape[0])
    for i in combo: v=v*x[:,i]
    return v
def build(x,spec):
    return np.column_stack([np.ones(x.shape[0])]+[mono(x,c) for c in spec])
def cde_terms(x):
    C,D,E=[x[:,idx[k]] for k in "CDE"]
    combos=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]
    return np.column_stack([np.ones(x.shape[0])]+[(C**c)*(D**d)*(E**e) for c,d,e in combos])

spec_d1=[(i,) for i in range(6)]
spec_d2=spec_d1+list(itertools.combinations(range(6),2))
spec_d3=spec_d2+list(itertools.combinations(range(6),3))
reps=[tuple(idx[k] for k in s) for s in ["ABC","ABD","ABE","ABF","ACD","ACE","ACF","ADE","ADF","AEF"]]

def loo_rmse(M,yy,lam=1e-6):
    p=M.shape[1]; P=lam*np.eye(p); P[0,0]=0
    A=np.linalg.inv(M.T@M+P); beta=A@(M.T@yy)
    h=np.einsum('ij,jk,ik->i',M,A,M)
    r=(yy-M@beta)/np.clip(1-h,1e-8,None)
    return np.sqrt(np.mean(r**2))

models={}
models['d1']=build(Xo,spec_d1)
models['d2']=build(Xo,spec_d2)
models['d3']=build(Xo,spec_d3)
ABF=[(idx[k],) for k in "ABF"]
models['CDE+ABF']=np.column_stack([cde_terms(Xo)]+[mono(Xo,c) for c in ABF])
models['CDE+ABF+pr']=np.column_stack([cde_terms(Xo)]+[mono(Xo,c) for c in ABF]+[mono(Xo,(idx[a],idx[b])) for a,b in ["AB","AF","BF"]])
models['d2+CDE']=build(Xo,spec_d2+[tuple(idx[k] for k in "CDE")])
models['d2+all3fi']=build(Xo,spec_d2+reps)
models['d3+CDExABF']=np.column_stack([build(Xo,spec_d3), np.column_stack([mono(Xo,(idx[k],))+0 for k in "ABF"])]) # placeholder
del models['d3+CDExABF']

print("LOO rmse within even half:")
for name,M in models.items():
    print(f"  {name:12s} p={M.shape[1]:2d} LOO {loo_rmse(M,yo):.4f}")

def gp_loo(theta,noise):
    D=(Xo[:,None,:]!=Xo[None,:,:]).sum(-1)/6.0
    K=np.exp(-theta*D)+noise*np.eye(len(obs)); Ki=np.linalg.inv(K)
    a=Ki@yo; return np.sqrt(np.mean((a/np.diag(Ki))**2))
best=None
for th in np.linspace(0.1,8,40):
    for nz in [1e-8,1e-6,1e-5,1e-4,1e-3]:
        s=gp_loo(th,nz)
        if best is None or s<best[0]: best=(s,th,nz)
print("GP hamming LOO",round(best[0],4),round(best[1],2),best[2])

# inspect CDE structure: values by (C,D,E) group with (A,B,F)
print("\nCDE groups (A B F -> utility):")
for c in [0,1]:
    for d in [0,1]:
        for e in [0,1]:
            sel=[i for i in obs if masks[i][2]==str(c) and masks[i][3]==str(d) and masks[i][4]==str(e)]
            strs=[(masks[i][0]+masks[i][1]+masks[i][5], round(float(y[i]),3)) for i in sel]
            print(f" C{c}D{d}E{e}:", strs)
