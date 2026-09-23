
import numpy as np, itertools
names=['A','B','C','D','E','F']
obs = {
"000000":0.9986468200270636,"111111":1.0,"000011":0.9906040268456375,
"000101":0.9932705248990579,"000110":0.9945945945945946,"001001":1.0,
"001010":0.558667676003028,"001100":1.0,"001111":1.0,"010001":1.0,
"010010":0.9879518072289156,"010100":0.9787798408488063,"010111":0.997289972899729,
"011000":1.0,"011011":0.7242394504416094,"011101":1.0,"011110":0.9986468200270636,
"100001":1.0,"100010":0.9986468200270636,"100100":1.0,"100111":1.0,
"101000":1.0,"101011":0.7178988326848249,"101101":1.0,"101110":1.0,
"110000":1.0,"110011":1.0,"110101":1.0,"110110":1.0,"111001":1.0,
"111010":0.5023825731790333,"111100":1.0}
masks=list(obs.keys()); y=np.array([obs[m] for m in masks])
X01=np.array([[int(c) for c in m] for m in masks])  # 0/1
Z=2*X01-1.0  # +/-1

def build(terms):
    cols=[np.ones(len(masks))]
    for t in terms:
        cols.append(np.prod(Z[:,t],axis=1))
    return np.array(cols).T

def fit_loo(cols,y):
    X=cols
    beta,_,_,_=np.linalg.lstsq(X,y,rcond=None)
    H=X@np.linalg.pinv(X.T@X)@X.T
    h=np.diag(H)
    resid=y-X@beta
    loo=np.sum((resid/(1-h))**2)/len(y)
    return beta,resid,np.sqrt(loo)

mains=[(i,) for i in range(6)]
pairs=[(i,j) for i in range(6) for j in range(i+1,6)]
triples=[t for t in itertools.combinations(range(6),3)]

for label,terms in [("order1",mains),("order2",mains+pairs),("order2+CDE",mains+pairs+[(2,3,4)]),
                    ("order2+CDE+CEF+DEF",mains+pairs+[(2,3,4),(2,4,5),(3,4,5)])]:
    beta,resid,loo=fit_loo(build(terms),y)
    print(label,"nparams",len(terms)+1,"LOO_rmse",round(loo,5),"maxabs_resid",round(np.max(np.abs(resid)),5))
    # show largest residuals
    idx=np.argsort(-np.abs(resid))[:6]
    for i in idx:
        print("   resid",round(resid[i],4),masks[i],[*terms])
