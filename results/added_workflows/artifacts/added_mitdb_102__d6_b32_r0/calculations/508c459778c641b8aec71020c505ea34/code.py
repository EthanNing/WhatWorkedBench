
import numpy as np, json
from itertools import combinations
from sklearn.linear_model import Lasso
obs=[("000000",0.9541284403669725),("111111",1.0),("000001",0.9771812080536912),("000010",0.9667994687915007),("000011",0.9837837837837838),("000100",0.9567567567567568),("000110",0.7906976744186046),("001000",0.9688385269121813),("001010",0.5536121673003802),("001101",0.8426073131955485),("010000",0.9732620320855615),("010111",0.7805695142378559),("011001",1.0),("011011",0.6953199617956065),("011100",0.9972451790633609),("011101",0.9972451790633609),("011110",0.8845686512758202),("100000",0.9667994687915007),("100101",1.0),("100111",0.8481012658227848),("101000",0.9688385269121813),("101001",0.9688385269121813),("101010",0.5536121673003802),("101011",0.6854990583804144),("101100",0.8426073131955485),("101101",0.8426073131955485),("101110",0.9904761904761905),("110000",0.9837837837837838),("110001",1.0),("110011",1.0),("110100",1.0),("110110",0.8389154704944178),("111000",1.0),("111010",0.6195744680851064)]
masks=[format(i,'06b') for i in range(64)]; midx={m:i for i,m in enumerate(masks)}
X=np.array([[int(c) for c in m] for m in masks],float); Z1=2*X-1
om=[midx[m] for m,u in obs]; y=np.array([u for m,u in obs]); n=len(y)
def designM(deg,Ms):
    cols=[np.ones(len(Ms))]
    for d in range(1,deg+1):
        for comb in combinations(range(6),d):
            c=np.ones(len(Ms))
            for j in comb: c*=Ms[:,j]
            cols.append(c)
    return np.column_stack(cols)
def lasso_loo(deg,al):
    Z=designM(deg,Z1)[om]; p=np.zeros(n)
    for i in range(n):
        idx=[j for j in range(n) if j!=i]
        m=Lasso(alpha=al,max_iter=200000).fit(Z[idx],y[idx]); p[i]=m.predict(Z[i].reshape(1,-1))[0]
    return np.sqrt(np.mean((p-y)**2))
for al in [0.0002,0.0005,0.001]:
    print("lasso deg3 full-LOO al",al, round(lasso_loo(3,al),4))
# choose alpha by inner LOO per fold for nested estimate (deg3), fixed small grid
grid=[0.0002,0.0005,0.001,0.002]
Z3=designM(3,Z1)[om]
p=np.zeros(n)
for i in range(n):
    idx=[j for j in range(n) if j!=i]
    # inner LOO pick alpha
    best=(9,None)
    for al in grid:
        pp=np.zeros(len(idx))
        for k,kk in enumerate(idx):
            sub=[j for j in idx if j!=kk]
            mm=Lasso(alpha=al,max_iter=200000).fit(Z3[sub],y[sub]); pp[k]=mm.predict(Z3[kk].reshape(1,-1))[0]
        rm=np.sqrt(np.mean((pp-y[idx])**2))
        if rm<best[0]: best=(rm,al)
    mm=Lasso(alpha=best[1],max_iter=200000).fit(Z3[idx],y[idx]); p[i]=mm.predict(Z3[i].reshape(1,-1))[0]
print("nested lasso deg3 rmse",round(np.sqrt(np.mean((p-y)**2)),4))
