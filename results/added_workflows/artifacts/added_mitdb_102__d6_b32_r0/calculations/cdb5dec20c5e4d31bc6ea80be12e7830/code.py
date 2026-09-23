
import numpy as np
from itertools import combinations
from sklearn.linear_model import LassoCV, Ridge
obs=[("000000",0.9541284403669725),("111111",1.0),("000001",0.9771812080536912),("000010",0.9667994687915007),("000011",0.9837837837837838),("000100",0.9567567567567568),("000110",0.7906976744186046),("001000",0.9688385269121813),("001010",0.5536121673003802),("001101",0.8426073131955485),("010000",0.9732620320855615),("010111",0.7805695142378559),("011001",1.0),("011011",0.6953199617956065),("011100",0.9972451790633609),("011101",0.9972451790633609),("011110",0.8845686512758202),("100000",0.9667994687915007),("100101",1.0),("100111",0.8481012658227848),("101000",0.9688385269121813),("101001",0.9688385269121813),("101010",0.5536121673003802),("101011",0.6854990583804144),("101100",0.8426073131955485),("101101",0.8426073131955485),("101110",0.9904761904761905),("110000",0.9837837837837838),("110001",1.0),("110011",1.0),("110100",1.0),("110110",0.8389154704944178),("111000",1.0),("111010",0.6195744680851064)]
masks=[format(i,'06b') for i in range(64)]; midx={m:i for i,m in enumerate(masks)}
X=np.array([[int(c) for c in m] for m in masks],float); Z1=2*X-1
om=[midx[m] for m,u in obs]; y=np.array([u for m,u in obs]); n=len(y)
def design(deg,Ms):
    cols=[np.ones(len(Ms))]
    for d in range(1,deg+1):
        for comb in combinations(range(6),d):
            cols.append(np.prod([Ms[:,j] for j in comb],axis=0))
    return np.column_stack(cols)
def loo_generic(fit_predict, Zfull, deg):
    Z=design(deg,Z1)[om]; p=np.zeros(n)
    for i in range(n):
        idx=[j for j in range(n) if j!=i]
        pred=fit_predict(Z[idx],y[idx],Z[i])
        p[i]=pred
    return p
def ridge_fp(al):
    def f(Ztr,ytr,z):
        A=Ztr.T@Ztr+al*np.eye(Ztr.shape[1]); b=np.linalg.solve(A,Ztr.T@ytr); return z@b
    return f
def lasso_fp(al):
    def f(Ztr,ytr,z):
        m=Lasso(alpha=al,max_iter=50000); m.fit(Ztr,ytr); return m.predict(z.reshape(1,-1))[0]
    return f
for deg in [2,3,4]:
    best=(9,None)
    for al in [0.0001,0.001,0.01,0.05,0.1,0.5,1,5,20,100,1000]:
        p=loo_generic(ridge_fp(al),Z1,deg); rm=np.sqrt(np.mean((p-y)**2))
        if rm<best[0]: best=(rm,al)
    print("ridge deg",deg, best)
from sklearn.linear_model import Lasso
for deg in [3,4]:
    best=(9,None)
    for al in [0.0005,0.001,0.002,0.005,0.01,0.02,0.05]:
        p=loo_generic(lasso_fp(al),Z1,deg); rm=np.sqrt(np.mean((p-y)**2))
        if rm<best[0]: best=(rm,al)
    print("lasso deg",deg, best)
