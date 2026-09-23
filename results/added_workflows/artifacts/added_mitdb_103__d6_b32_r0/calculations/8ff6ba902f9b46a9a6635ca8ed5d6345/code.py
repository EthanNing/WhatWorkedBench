
import numpy as np, itertools
masks=[format(i,'06b') for i in range(64)]
X=np.array([[1 if c=='1' else -1 for c in m] for m in masks],float)
y=np.full(64,np.nan)
for o in observations: y[masks.index(o['mask'])]=o['utility']
obs=np.where(np.isfinite(y))[0]
idx={c:i for i,c in enumerate("ABCDEF")}

# per-CDE-group exact additive solve: y = m + aA + bB + fF
print("group : m(base at A=B=F=-1)  a  b  f   [along observed: value]")
tab={}
for c in [0,1]:
  for d in [0,1]:
    for e in [0,1]:
      sel=[i for i in obs if masks[i][2]==str(c) and masks[i][3]==str(d) and masks[i][4]==str(e)]
      M=np.array([[1, X[i,0],X[i,1],X[i,5]] for i in sel])
      yy=np.array([y[i] for i in sel])
      sol=np.linalg.solve(M,yy)
      tab[(c,d,e)]=sol
      print(f"C{c}D{d}E{e}: m={sol[0]: .3f} a={sol[1]: .3f} b={sol[2]: .3f} f={sol[3]: .3f}",
            [(masks[i][0]+masks[i][1]+masks[i][5],round(float(y[i]),3)) for i in sel])
print()
# pooled model residual
Xd=np.column_stack([X[:,2],X[:,3],X[:,4],X[:,2]*X[:,3],X[:,2]*X[:,4],X[:,3]*X[:,4],X[:,2]*X[:,3]*X[:,4]])
def design(x):
    return np.column_stack([np.ones(len(x)), x[:,2],x[:,3],x[:,4],x[:,2]*x[:,3],x[:,2]*x[:,4],x[:,3]*x[:,4],x[:,2]*x[:,3]*x[:,4], x[:,0],x[:,1],x[:,5]])
M=design(X[obs]); beta,_,_,_=np.linalg.lstsq(M,y[obs],rcond=None)
r=y[obs]-M@beta
print("pooled CDE+ABF residual rmse:",np.sqrt(np.mean(r**2)),"max",np.max(np.abs(r)))
print("worst:",[(masks[obs[k]],round(float(r[k]),3)) for k in np.argsort(-np.abs(r))[:6]])
# pooled A,B,F coeffs
print("coeffs a,b,f:",np.round(beta[8:11],4))
