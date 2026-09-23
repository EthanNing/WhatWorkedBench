
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
    x=bits(m); t=[1.0]; 
    if kind in ('d1','d2','d3','d2p'): t+=list(x)
    if kind in ('d2','d3','d2p'):
        for i in range(6):
            for j in range(i+1,6): t.append(x[i]*x[j])
    if kind=='d3':
        for i in range(6):
            for j in range(i+1,6):
                for k in range(j+1,6): t.append(x[i]*x[j]*x[k])
    if kind=='d2p':  # deg2 + F*pair(A-E)
        for i in range(5):
            for j in range(i+1,5): t.append(x[5]*x[i]*x[j])
    return np.array(t)
def ridge_loo(kind,alphas):
    X=np.array([basis(m,kind) for m in masks]); n=len(y); p=X.shape[1]
    best=None
    for a in alphas:
        err=0.0
        for i in range(n):
            tr=np.arange(n)!=i
            w=np.linalg.solve(X[tr].T@X[tr]+a*np.eye(p), X[tr].T@y[tr])
            err+=(X[i]@w-y[i])**2
        r=np.sqrt(err/n)
        if best is None or r<best[0]: best=(r,a)
    a=best[1]; w=np.linalg.solve(X.T@X+a*np.eye(p),X.T@y)
    return best[0],a,w,X
alphas=np.logspace(-5,4,37)
res={}
for kind in ['d1','d2','d2p','d3']:
    r,a,w,X=ridge_loo(kind,alphas)
    res[kind]=(r,a,w,X); print("%-4s LOO rms %.5f  alpha %.4g  p=%d"%(kind,r,a,X.shape[1]))
# k-NN in hamming
def knn_loo(k):
    err=0
    M=np.array([bits(m) for m in masks])
    for i in range(len(y)):
        d=np.abs(M-M[i]).sum(1); d[i]=99
        nb=np.argsort(d)[:k]; err+=(y[nb].mean()-y[i])**2
    return np.sqrt(err/len(y))
for k in [1,2,3,4,5]:
    print("knn k=%d LOO rms %.5f"%(k,knn_loo(k)))
