
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
def cde_cols(x):
    C,D,E=[x[:,idx[k]] for k in "CDE"]
    return [(C**c)*(D**d)*(E**e) for c,d,e in [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]]
def D(x,extras): return np.column_stack([np.ones(x.shape[0])]+cde_cols(x)+[mono(x,c) for c in extras])
common=[(idx['A'],),(idx['B'],),(idx['F'],)]
pairBF=[(idx['B'],idx['F'])]
sets={
 'CDE+ABF':common,
 'CDE+ABF+BF':common+pairBF,
 'CDE+ABF+ABFpairs':common+[(idx['A'],idx['B']),(idx['A'],idx['F']),pairBF[0]],
 'CDE+ABF+BFinA':common+pairBF+[(idx['A'],idx['B'],idx['F'])],
}

def loo_pred(M,yy):
    A=np.linalg.inv(M.T@M+1e-8*np.eye(M.shape[1]))
    b=A@(M.T@yy); h=np.einsum('ij,jk,ik->i',M,A,M)
    return (yy-h*yy-(M@b-h*yy))/1  # placeholder
def loo(M,yy):
    A=np.linalg.inv(M.T@M+1e-8*np.eye(M.shape[1]))
    b=A@(M.T@yy); h=np.einsum('ij,jk,ik->i',M,A,M)
    pred=M@b
    return (yy-h*yy - (pred-h*yy))/(1-h)  # = (pred - h*yy)/(1-h)

trans={'id':(lambda v:v,lambda v:v),
 'deficit':(lambda v:1-v,lambda v:1-v),
 'inv':(lambda v:1/np.clip(v,1e-3,1.),lambda v:1/np.clip(v,1e-3,None)),
 'logit':(lambda v:np.log(np.clip(v,1e-3,1-1e-3)/(1-np.clip(v,1e-3,1-1e-3))),
          lambda v:1/(1+np.exp(-v)))}
for sname,extras in sets.items():
    M=D(Xo,extras); row=[]
    for tname,(f,finv) in trans.items():
        z=f(yo); p=loo(M,z); yhat=np.clip(finv(p),0,1)
        row.append(f"{tname}={np.sqrt(np.mean((yhat-yo)**2)):.4f}")
    print(f"{sname:20s} p={M.shape[1]:2d} "+"  ".join(row))
