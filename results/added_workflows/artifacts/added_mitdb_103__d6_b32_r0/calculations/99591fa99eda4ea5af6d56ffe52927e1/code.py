
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
CDE=[(idx['C'],),(idx['D'],),(idx['E'],),(idx['C'],idx['D']),(idx['C'],idx['E']),(idx['D'],idx['E']),(idx['C'],idx['D'],idx['E'])]
A=(idx['A'],);B=(idx['B'],);F=(idx['F'],);BF=(idx['B'],idx['F'])
def dm(x,extras): return np.column_stack([np.ones(x.shape[0])]+[mono(x,c) for c in CDE+extras])
models={
 'CDE+ABF':dm(Xo,[A,B,F]),
 'CDE+ABF+BF':dm(Xo,[A,B,F,BF]),
 'CDE+A+BF+ABF':dm(Xo,[A,B,F,BF,(idx['A'],idx['B'],idx['F'])]),
 'CDE+A+B+F+AB+AF+BF':dm(Xo,[A,B,F,(idx['A'],idx['B']),(idx['A'],idx['F']),BF]),
 'CDE+ABFfull':dm(Xo,[A,B,F,(idx['A'],idx['B']),(idx['A'],idx['F']),BF,(idx['A'],idx['B'],idx['F'])]),
}
def loo(M,yy):
    A_=np.linalg.inv(M.T@M+1e-9*np.eye(M.shape[1])); b=A_@(M.T@yy)
    h=np.einsum('ij,jk,ik->i',M,A_,M)
    r=(yy-M@b)/np.clip(1-h,1e-6,None); return np.sqrt(np.mean(r**2)),b
print("LOO (34 pts):")
best=None
for k,M in models.items():
    s,b=loo(M,yo); print(f"  {k:22s} p={M.shape[1]:2d} LOO {s:.4f}")
    if best is None or s<best[0]: best=(s,k,b,M)
print("best:",best[1],round(best[0],4))
_,kname,_,_=best
Mfull=dm(X,[A,B,F,BF]) if 'BF' in kname else dm(X,[A,B,F])
# report per-group implied structure using observed
print("\nper-group observed (A,B,F)->y:")
for c in [0,1]:
 for d in [0,1]:
  for e in [0,1]:
   sel=[i for i in obs if masks[i][2]==str(c) and masks[i][3]==str(d) and masks[i][4]==str(e)]
   print(f"C{c}D{d}E{e}:",sorted([(masks[i][0]+masks[i][1]+masks[i][5],round(float(y[i]),3)) for i in sel]))
