
import numpy as np
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
np.random.seed(0)
obs=[("000000",0.9541284403669725),("111111",1.0),("000001",0.9771812080536912),("000010",0.9667994687915007),("000011",0.9837837837837838),("000100",0.9567567567567568),("000110",0.7906976744186046),("001000",0.9688385269121813),("001010",0.5536121673003802),("001101",0.8426073131955485),("010000",0.9732620320855615),("010111",0.7805695142378559),("011001",1.0),("011011",0.6953199617956065),("011100",0.9972451790633609),("011101",0.9972451790633609),("011110",0.8845686512758202),("100000",0.9667994687915007),("100101",1.0),("100111",0.8481012658227848),("101000",0.9688385269121813),("101001",0.9688385269121813),("101010",0.5536121673003802),("101011",0.6854990583804144),("101100",0.8426073131955485),("101101",0.8426073131955485),("101110",0.9904761904761905),("110000",0.9837837837837838),("110001",1.0),("110011",1.0),("110100",1.0),("110110",0.8389154704944178),("111000",1.0),("111010",0.6195744680851064)]
masks=[format(i,'06b') for i in range(64)]; midx={m:i for i,m in enumerate(masks)}
X=np.array([[int(c) for c in m] for m in masks],float); Z1=2*X-1
om=[midx[m] for m,u in obs]; y=np.array([u for m,u in obs]); n=len(y)
Zc=Z1[om]
def loo_preds(fitpred):
    p=np.zeros(n)
    for i in range(n):
        idx=[j for j in range(n) if j!=i]
        p[i]=fitpred(Zc[idx],y[idx],Zc[i])
    return np.sqrt(np.mean((p-y)**2)),p
# RF
for ntree in [300,800]:
    fp=lambda A,b,z,nt=ntree: RandomForestRegressor(n_estimators=nt,random_state=0,min_samples_leaf=1).fit(A,b).predict(z.reshape(1,-1))[0]
    print("RF",ntree, round(loo_preds(fp)[0],4))
for lr,ne,md in [(0.05,200,1),(0.03,500,1),(0.1,200,2)]:
    fp=lambda A,b,z,lr=lr,ne=ne,md=md: GradientBoostingRegressor(learning_rate=lr,n_estimators=ne,max_depth=md,random_state=0).fit(A,b).predict(z.reshape(1,-1))[0]
    print("GBM",lr,ne,md, round(loo_preds(fp)[0],4))
# poly kernel ridge degree 3
def polyk(A,B,d,coef=1.0):
    return (coef+ (A@B.T)/6.0)**d
for d in [2,3]:
    for lam in [0.001,0.01,0.03,0.1,0.3,1.0]:
        def fp(A,b,z,d=d,lam=lam):
            K=polyk(A,A,d); Kz=polyk(z.reshape(1,-1),A,d).ravel()
            al=np.linalg.solve(K+lam*np.eye(len(A)),b); return Kz@al
        print("polyK",d,lam, round(loo_preds(fp)[0],4))
