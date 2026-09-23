
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

def build(x,spec):  # spec list of combos
    cols=[np.ones(x.shape[0])]
    for combo in spec:
        if combo: cols.append(mono(x,combo))
    return np.column_stack(cols)

spec_d1=[(i,) for i in range(6)]
spec_d2=spec_d1+[c for c in itertools.combinations(range(6),2)]
spec_d3=spec_d2+[c for c in itertools.combinations(range(6),3)]

def loo_rmse(M,yy,lam=1e-6):
    p=M.shape[1]; P=lam*np.eye(p); P[0,0]=0
    A=np.linalg.inv(M.T@M+P)
    beta=A@(M.T@yy); h=np.einsum('ij,jk,ik->i',M,A,M)
    r=(yy-M@beta)/np.clip(1-h,1e-8,None)
    return np.sqrt(np.mean(r**2)), beta, A

# CDE blocks
CDE=list(itertools.product([0,1],repeat=3))  # values for (C,D,E)
def cde_terms(x):
    C,D,E=[x[:,idx[k]] for k in "CDE"]  # ±1
    cols=[np.ones(x.shape[0])]
    for (c,d,e) in [ (1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]:
        v=(C**c)*(D**d)*(E**e); cols.append(v)
    return np.column_stack(cols)

models={}
M,_=build(Xo,[]); models['null']=M
M,_=build(Xo,spec_d1); models['d1']=M
M,_=build(Xo,spec_d2); models['d2']=M
M,_=build(Xo,spec_d3); models['d3']=M
# CDE + A,B,F mains
M=np.column_stack([cde_terms(Xo)]+[mono(Xo,(idx[k],)) for k in "ABF"]); models['CDE+ABF']=M
M=np.column_stack([cde_terms(Xo)]+[mono(Xo,(idx[k],)) for k in "ABF"]+[mono(Xo,(idx[a],idx[b])) for a,b in ["AB","AF","BF"]]); models['CDE+ABF+pairs']=M
# degree2 + CDE 3FI
M=build(Xo,spec_d2+[(idx['C'],idx['D'],idx['E'])]); models['d2+CDE']=M
# degree2 + all 10 identifiable 3fi representatives
reps=[(idx[a],idx[b],idx[c]) for a,b,c in ["ABC","ABD","ABE","ABF","ACD","ACE","ACF","ADE","ADF","AEF"]]
M=build(Xo,spec_d2+reps); models['d2+all3fi']=M

print("LOO rmse within even half:")
for name,M in models.items():
    s,_,_=loo_rmse(M,yo); print(f"  {name:16s} p={M.shape[1]:2d} LOO {s:.4f}")

# GP
def gp_loo(theta,noise):
    D=(Xo[:,None,:]!=Xo[None,:,:]).sum(-1)/6.0
    K=np.exp(-theta*D)+noise*np.eye(len(obs)); Ki=np.linalg.inv(K)
    a=Ki@yo; return np.sqrt(np.mean((a/np.diag(Ki))**2))
best=None
for th in np.linspace(0.1,8,40):
    for nz in [1e-8,1e-6,1e-5,1e-4,1e-3]:
        s=gp_loo(th,nz)
        if best is None or s<best[0]: best=(s,th,nz)
print("GP hamming LOO",round(best[0],4),best[1],best[2])
