
import numpy as np, itertools
obs=dict([("000000",0.5),("111111",0.74951171875),("000010",0.69482421875),("000100",0.759765625),
("000111",0.75830078125),("001000",0.759765625),("001011",0.75830078125),("001100",0.759765625),
("001101",0.759765625),("010000",0.75341796875),("010100",0.76416015625),("010101",0.75830078125),
("010110",0.73583984375),("010111",0.75830078125),("011000",0.75830078125),("011001",0.75830078125),
("011011",0.76123046875),("011110",0.75732421875),("011111",0.76123046875),("100000",0.755859375),
("100011",0.76123046875),("100101",0.751953125),("100110",0.76220703125),("101000",0.751953125),
("101010",0.76025390625),("101011",0.75439453125),("101101",0.7509765625),("101111",0.75439453125),
("110001",0.75634765625),("110110",0.76123046875),("111001",0.75634765625),("111010",0.76123046875),
("111100",0.75244140625),("111110",0.76220703125)])
singles=['100000','010000','001000','000100','000010']
obs['000001']=0.5
for s in singles: obs[s[:5]+str(1-int(s[5]))]=obs[s]
masks=list(obs.keys()); y=np.array([obs[m] for m in masks])
def bits(m): return np.array([int(c) for c in m],float)
def basis(m,kind):
    x=bits(m); t=[1.0]
    if kind!='n': t+=list(x)
    if kind in ('d2','d3'):
        for i in range(6):
            for j in range(i+1,6): t.append(x[i]*x[j])
    if kind=='d3':
        for i in range(6):
            for j in range(i+1,6):
                for k in range(j+1,6): t.append(x[i]*x[j]*x[k])
    return np.array(t)
popAE=np.array([sum(int(c) for c in m[:5]) for m in masks])
mask_hi=popAE>=2  # normal region
def ridge_loo(kind,alphas,sel=None):
    if sel is None: sel=np.ones(len(y),bool)
    X=np.array([basis(m,kind) for m in masks]); n=len(y); p=X.shape[1]
    yy=y.copy(); errs=np.zeros(n)
    best=None
    for a in alphas:
        e=0.0; ec=0
        for i in range(n):
            if not sel[i]: continue
            tr=(np.arange(n)!=i)
            w=np.linalg.solve(X[tr].T@X[tr]+a*np.eye(p), X[tr].T@y[tr])
            e+=(X[i]@w-y[i])**2; ec+=1
        r=np.sqrt(e/ec)
        if best is None or r<best[0]: best=(r,a)
    a=best[1]
    for i in range(n):
        if not sel[i]: continue
        tr=(np.arange(n)!=i)
        w=np.linalg.solve(X[tr].T@X[tr]+a*np.eye(p), X[tr].T@y[tr])
        errs[i]=X[i]@w-y[i]
    w=np.linalg.solve(X.T@X+a*np.eye(p),X.T@y)
    return best[0],a,w,errs
alphas=np.logspace(-5,4,37)
print("Restricted LOO (popcount>=2 cells):")
for kind in ['n','d1','d2','d3']:
    r,a,w,errs=ridge_loo(kind,alphas,mask_hi)
    print("%-3s rms %.5f alpha %.4g"%(kind,r,a))
# kNN restricted
M=np.array([bits(m) for m in masks])
for k in [2,3,4,5,6]:
    sub=np.where(mask_hi)[0]; e=0
    for i in sub:
        d=np.abs(M-M[i]).sum(1); d[i]=99
        nb=np.argsort(d)[:k]; e+=(y[nb].mean()-y[i])**2
    print("knn k=%d restricted rms %.5f"%(k,np.sqrt(e/len(sub))))
# per-cell LOO errors for d2 and d3 on all cells
for kind in ['d2','d3']:
    r,a,w,errs=ridge_loo(kind,alphas)
    big=sorted(range(len(y)),key=lambda i:-abs(errs[i]))[:8]
    print(kind,"worst:",[(masks[i],round(errs[i],4)) for i in big])
