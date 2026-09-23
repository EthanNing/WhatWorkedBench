
import numpy as np
from itertools import combinations
obs=[("000000",0.9541284403669725),("111111",1.0),("000001",0.9771812080536912),("000010",0.9667994687915007),("000011",0.9837837837837838),("000100",0.9567567567567568),("000110",0.7906976744186046),("001000",0.9688385269121813),("001010",0.5536121673003802),("001101",0.8426073131955485),("010000",0.9732620320855615),("010111",0.7805695142378559),("011001",1.0),("011011",0.6953199617956065),("011100",0.9972451790633609),("011101",0.9972451790633609),("011110",0.8845686512758202),("100000",0.9667994687915007),("100101",1.0),("100111",0.8481012658227848),("101000",0.9688385269121813),("101001",0.9688385269121813),("101010",0.5536121673003802),("101011",0.6854990583804144),("101100",0.8426073131955485),("101101",0.8426073131955485),("101110",0.9904761904761905),("110000",0.9837837837837838),("110001",1.0),("110011",1.0),("110100",1.0),("110110",0.8389154704944178),("111000",1.0),("111010",0.6195744680851064)]
masks=[format(i,'06b') for i in range(64)]; midx={m:i for i,m in enumerate(masks)}
X=np.array([[int(c) for c in m] for m in masks],float); Z1=2*X-1
om=[midx[m] for m,u in obs]; y=np.array([u for m,u in obs]); n=len(y)
allterms=[]
for d in range(1,7):
    for comb in combinations(range(6),d): allterms.append(comb)
def col(comb,Ms):
    c=np.ones(len(Ms))
    for j in comb: c*=Ms[:,j]
    return c
Zfull=np.column_stack([np.ones(n)]+[col(c,Z1[om]) for c in allterms])  # n x 64
def loo_rmse(cols_idx):
    A=Zfull[:,cols_idx]
    b,_,_,_=np.linalg.lstsq(A,y,rcond=None)
    H=A@np.linalg.pinv(A.T@A)@A.T
    h=np.clip(np.diag(H),0,0.999)
    res=(y-A@b)/(1-h)
    return np.sqrt(np.mean(res**2))
sel=[0]
cur=loo_rmse(sel); print("start LOO",round(cur,4))
cand=list(range(1,64))
for step in range(20):
    best=None;bv=cur
    for c in cand:
        v=loo_rmse(sel+[c])
        if v<bv-1e-6: bv=v;best=c
    if best is None: break
    sel.append(best); cand.remove(best); cur=bv
    print("add",allterms[best-1],"LOO",round(cur,4))
print("final terms",[allterms[i-1] for i in sel[1:]])
