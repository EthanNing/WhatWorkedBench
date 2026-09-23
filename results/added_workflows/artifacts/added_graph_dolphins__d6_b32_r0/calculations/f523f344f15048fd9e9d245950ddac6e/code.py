
import numpy as np, itertools
from wwb_helpers import fit_surface
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
def bits(m): return np.array([int(c) for c in m],float)
def pop(m): return sum(int(c) for c in m[:5])
allm=[''.join(map(str,b)) for b in itertools.product([0,1],repeat=6)]
allK=list(obs.keys()); allM=np.array([bits(m) for m in allK]); allY=np.array([obs[m] for m in allK])
normal=[i for i,m in enumerate(allK) if pop(m)>=2]
# Lasso with all interactions up to deg2/3, standardized, LOO
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
def basis(m,deg):
    x=bits(m); t=list(x)
    if deg>=2:
        for a in range(6):
            for b in range(a+1,6): t.append(x[a]*x[b])
    if deg>=3:
        for a in range(6):
            for b in range(a+1,6):
                for c in range(b+1,6): t.append(x[a]*x[b]*x[c])
    return np.array(t)
for deg in [2,3]:
    Xall=np.array([basis(m,deg) for m in allm])
    for al in [0.001,0.003,0.01,0.03]:
        errs=[]
        for i in normal:
            tr=[j for j in normal if j!=i]
            mdl=Lasso(alpha=al,max_iter=5000).fit(Xall[[allm.index(allK[j]) for j in tr]],allY[tr])
            errs.append(mdl.predict(Xall[[allm.index(allK[i])]])[0]-allY[i])
        print("lasso deg%d a=%.3f rms %.5f"%(deg,al,np.sqrt(np.mean(np.square(errs)))))
for n_est,md in [(300,3),(300,6)]:
    errs=[]
    for i in normal:
        tr=[j for j in normal if j!=i]
        mdl=RandomForestRegressor(n_estimators=n_est,max_depth=md,random_state=0).fit(np.array([bits(allK[j]) for j in tr]),allY[tr])
        errs.append(mdl.predict(bits(allK[i]).reshape(1,-1))[0]-allY[i])
    print("RF d%d rms %.5f"%(md,np.sqrt(np.mean(np.square(errs)))))
# helper fit_surface predictions
for d in [1,2]:
    pred=fit_surface(observations,degree=d)
    pr=dict(zip(allm,pred)) if isinstance(pred,dict) else None
    if pr is None:
        pr=dict(zip(allm,np.asarray(pred)))
    diffs=[abs(pr[m]-obs[m]) for m in allK]
    print("fit_surface deg%d: mean abs err on known %.5f  unobs preds:"%(d,np.mean(diffs)),
          {m:round(pr[m],4) for m in allm if m not in obs})
