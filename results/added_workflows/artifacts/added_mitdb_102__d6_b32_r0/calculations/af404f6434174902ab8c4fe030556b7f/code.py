
import numpy as np
from itertools import combinations
from sklearn.linear_model import Lasso, ElasticNet, LassoLars
obs=[("000000",0.9541284403669725),("111111",1.0),("000001",0.9771812080536912),("000010",0.9667994687915007),("000011",0.9837837837837838),("000100",0.9567567567567568),("000110",0.7906976744186046),("001000",0.9688385269121813),("001010",0.5536121673003802),("001101",0.8426073131955485),("010000",0.9732620320855615),("010111",0.7805695142378559),("011001",1.0),("011011",0.6953199617956065),("011100",0.9972451790633609),("011101",0.9972451790633609),("011110",0.8845686512758202),("100000",0.9667994687915007),("100101",1.0),("100111",0.8481012658227848),("101000",0.9688385269121813),("101001",0.9688385269121813),("101010",0.5536121673003802),("101011",0.6854990583804144),("101100",0.8426073131955485),("101101",0.8426073131955485),("101110",0.9904761904761905),("110000",0.9837837837837838),("110001",1.0),("110011",1.0),("110100",1.0),("110110",0.8389154704944178),("111000",1.0),("111010",0.6195744680851064)]
masks=[format(i,'06b') for i in range(64)]; midx={m:i for i,m in enumerate(masks)}
X=np.array([[int(c) for c in m] for m in masks],float); Z1=2*X-1
om=[midx[m] for m,u in obs]; y=np.array([u for m,u in obs]); n=len(y)
def terms(deg):
    T=[]
    for d in range(1,deg+1):
        for comb in combinations(range(6),d): T.append(comb)
    return T
def designM(deg,Ms):
    T=terms(deg); cols=[np.ones(len(Ms))]
    for comb in T: cols.append(np.prod([Ms[:,j] for j in comb],axis=0))
    return np.column_stack(cols)
def loo_lasso(deg,al,l1r=1.0,maxit=100000):
    Z=designM(deg,Z1)[om]; p=np.zeros(n)
    for i in range(n):
        idx=[j for j in range(n) if j!=i]
        if l1r==1.0: m=Lasso(alpha=al,max_iter=maxit)
        else: m=ElasticNet(alpha=al,l1_ratio=l1r,max_iter=maxit)
        m.fit(Z[idx],y[idx]); p[i]=m.predict(Z[i].reshape(1,-1))[0]
    return np.sqrt(np.mean((p-y)**2)),p
best=(9,)
for deg in [3,4,5]:
    for al in [0.0002,0.0003,0.0005,0.0007,0.001,0.0015,0.002,0.003]:
        rm,p=loo_lasso(deg,al)
        if rm<best[0]: best=(rm,deg,al)
        print("lasso deg",deg,"al",al,"loo",round(rm,4))
print("BEST",best)
rm,p=loo_lasso(best[1],best[2])
print("best loo preds vs y:")
for i in range(n): print(obs[i][0], round(y[i],4), round(p[i],4))
