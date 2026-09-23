
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
"111010":0.5023825731790333,"111100":1.0,"011010":0.5023825731790333,"101010":0.558667676003028}

def bits(m): return [int(c) for c in m]
c0even=[m for m in [format(i,'06b') for i in range(64)] if m[2]=='0' and sum(bits(m))%2==0]
c0odd =[m for m in [format(i,'06b') for i in range(64)] if m[2]=='0' and sum(bits(m))%2==1]
sub=[0,1,3,4,5]; sn=['A','B','D','E','F']
def Z(ms): return np.array([[1 if m[i]=='1' else -1 for i in sub] for m in ms])
Ze,Zo=Z(c0even),Z(c0odd)
e=np.array([round(738*(1-obs[m])/obs[m]) for m in c0even],float)
allterms=[(j,) for j in range(5)]+[(i,j) for i in range(5) for j in range(i+1,5)]
def design(Zsel,terms): 
    c=[np.ones(len(Zsel))]+[np.prod(Zsel[:,list(t)],axis=1) for t in terms]; return np.array(c).T
def loo_rmse(terms):
    X=design(Ze,terms); 
    H=X@np.linalg.pinv(X.T@X)@X.T; h=np.diag(H)
    r=e-X@np.linalg.lstsq(X,e,rcond=None)[0]
    return np.sqrt(np.mean((r/(1-h))**2))
# forward selection
sel=[]; best=loo_rmse([])
print("start LOO",round(best,3))
for _ in range(10):
    cand=[(loo_rmse(sel+[t]),t) for t in allterms if t not in sel]
    cand.sort(); 
    if cand[0][0]<best-1e-6: best,t=cand[0]; sel.append(t); print("add",[sn[i] for i in t],"LOO",round(best,3))
    else: break
print("selected:",[[sn[i] for i in t] for t in sel])
X=design(Ze,sel); coef=np.linalg.lstsq(X,e,rcond=None)[0]
ehat_o=design(Zo,sel)@coef
print("\nodd C=0 predictions:")
for m,v in zip(c0odd,ehat_o): print("  ",m,f"e={v:6.2f} u={738/(738+max(v,0)):.5f}")
