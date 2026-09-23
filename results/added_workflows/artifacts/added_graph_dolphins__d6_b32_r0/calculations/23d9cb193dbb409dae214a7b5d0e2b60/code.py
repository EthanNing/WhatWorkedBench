
import numpy as np, itertools
obs34=[("000000",0.5),("111111",0.74951171875),("000010",0.69482421875),("000100",0.759765625),
("000111",0.75830078125),("001000",0.759765625),("001011",0.75830078125),("001100",0.759765625),
("001101",0.759765625),("010000",0.75341796875),("010100",0.76416015625),("010101",0.75830078125),
("010110",0.73583984375),("010111",0.75830078125),("011000",0.75830078125),("011001",0.75830078125),
("011011",0.76123046875),("011110",0.75732421875),("011111",0.76123046875),("100000",0.755859375),
("100011",0.76123046875),("100101",0.751953125),("100110",0.76220703125),("101000",0.751953125),
("101010",0.76025390625),("101011",0.75439453125),("101101",0.7509765625),("101111",0.75439453125),
("110001",0.75634765625),("110110",0.76123046875),("111001",0.75634765625),("111010",0.76123046875),
("111100",0.75244140625),("111110",0.76220703125)]
O=dict(obs34)
# deduced exact cells (F-invariance for <=1 active feature among A-E)
singles=['100000','010000','001000','000100','000010']
ded={'000001':0.5}
for s in singles: ded[s[:5]+str(1-int(s[5]))]=O[s]
print("deduced:",ded)
obs=dict(O); obs.update(ded)
print("n exact cells:",len(obs))
allm=[''.join(map(str,b)) for b in itertools.product([0,1],repeat=6)]
idx={m:i for i,m in enumerate(allm)}
unobs=[m for m in allm if m not in obs]
print("unobserved (%d):"%len(unobs),unobs)

def Xb(m,deg):
    x=np.array([int(c) for c in m],float)
    t=[1.0]; t+=list(x)
    if deg>=2:
        for i in range(6):
            for j in range(i+1,6): t.append(x[i]*x[j])
    if deg>=3:
        for i in range(6):
            for j in range(i+1,6):
                for k in range(j+1,6): t.append(x[i]*x[j]*x[k])
    return np.array(t)
masks=[m for m,v in obs.items()]; y=np.array([v for m,v in obs.items()])

def loo_ridge(X,y,alphas):
    n=len(y); best=None
    for a in alphas:
        err=0
        for i in range(n):
            tr=[j for j in range(n) if j!=i]
            Xt=X[tr]; yt=y[tr]
            A=Xt.T@Xt+a*np.eye(X.shape[1])
            w=np.linalg.solve(A,Xt.T@yt)
            p=X[i]@w
            err+=(p-y[i])**2
        rms=np.sqrt(err/n)
        if best is None or rms<best[0]: best=(rms,a)
    A=X.T@X+best[1]*np.eye(X.shape[1]); w=np.linalg.solve(A,X.T@y)
    return best,w
alphas=np.logspace(-4,3,29)
for deg in [1,2,3]:
    X=np.array([Xb(m,deg) for m in masks])
    rms,w=loo_ridge(X,y,alphas)
    print("deg",deg,"LOO rms %.5f alpha %.4g"%(rms,w[0]*0+ (w[0])))
