
import numpy as np
from itertools import combinations
obs=[("000000",0.9541284403669725),("111111",1.0),("000001",0.9771812080536912),("000010",0.9667994687915007),("000011",0.9837837837837838),("000100",0.9567567567567568),("000110",0.7906976744186046),("001000",0.9688385269121813),("001010",0.5536121673003802),("001101",0.8426073131955485),("010000",0.9732620320855615),("010111",0.7805695142378559),("011001",1.0),("011011",0.6953199617956065),("011100",0.9972451790633609),("011101",0.9972451790633609),("011110",0.8845686512758202),("100000",0.9667994687915007),("100101",1.0),("100111",0.8481012658227848),("101000",0.9688385269121813),("101001",0.9688385269121813),("101010",0.5536121673003802),("101011",0.6854990583804144),("101100",0.8426073131955485),("101101",0.8426073131955485),("101110",0.9904761904761905),("110000",0.9837837837837838),("110001",1.0),("110011",1.0),("110100",1.0),("110110",0.8389154704944178),("111000",1.0),("111010",0.6195744680851064)]
masks=[format(i,'06b') for i in range(64)]; midx={m:i for i,m in enumerate(masks)}
X=np.array([[int(c) for c in m] for m in masks],float); Z1=2*X-1
om=[midx[m] for m,u in obs]; y=np.array([u for m,u in obs]); n=len(y)
allterms=[]
for d in range(1,7):
    for comb in combinations(range(6),d): allterms.append(comb)
def colv(comb,Ms):
    c=np.ones(len(Ms))
    for j in comb: c*=Ms[:,j]
    return c
Zfull=np.column_stack([np.ones(n)]+[colv(c,Z1[om]) for c in allterms]])
def loo(A,yv):
    b,_,_,_=np.linalg.lstsq(A,yv,rcond=None)
    G=np.linalg.pinv(A.T@A); h=np.clip(np.einsum('ij,jk,ik->i',A,G,A),0,0.999)
    return np.sqrt(np.mean(((yv-A@b)/(1-h))**2))
def forward(rows,yv,maxterms=18):
    Av=Zfull[rows].copy(); sel=[0]; cand=list(range(1,64)); cur=loo(Av[:,[0]],yv)
    for step in range(maxterms):
        best=None;bv=cur
        for c in cand:
            v=loo(Av[:,sel+[c]],yv)
            if v<bv-1e-7: bv=v;best=c
        if best is None: break
        sel.append(best); cand.remove(best); cur=bv
    return sel
preds=np.zeros(n)
for i in range(n):
    rows=[j for j in range(n) if j!=i]
    sel=forward(rows,y[rows],18)
    b,_,_,_=np.linalg.lstsq(Zfull[np.ix_(rows,sel)],y[rows],rcond=None)
    preds[i]=Zfull[i,sel]@b
print("nested-CV LOO rmse",round(np.sqrt(np.mean((preds-y)**2)),4))
for i in range(n): print(obs[i][0], round(y[i],4), round(preds[i],4))
