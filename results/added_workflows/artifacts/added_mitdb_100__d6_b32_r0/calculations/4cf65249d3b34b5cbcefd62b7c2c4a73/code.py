
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
# verify integer error model u=738/(738+e)
for m,u in obs.items():
    e=738*(1-u)/u
    assert abs(e-round(e))<0.02,(m,u,e)
print("all observed consistent with u=738/(738+e), integer e. good")

def bits(m): return [int(c) for c in m]
# ---- C=0 error model: mains + 2fi exact on 16 even C=0 points ----
c0masks=[m for m in [format(i,'06b') for i in range(64)] if m[2]=='0' and (sum(bits(m))%2==0)]
print("C=0 even masks:",len(c0masks))
sub_facs=[0,1,3,4,5]  # A,B,D,E,F
Zsub=np.array([[1 if m[i]=='1' else -1 for i in sub_facs] for m in c0masks])  # 16x5
eterms=lambda M: [round(738*(1-obs[m])/obs[m]) for m in M] if False else None
evec=np.array([round(738*(1-obs[m])/obs[m]) for m in c0masks])
print("e (even C=0):",dict(zip(c0masks,evec)))
# basis: intercept, mains, 2fi
subnames=['A','B','D','E','F']
cols=[np.ones(16)]+[Zsub[:,j] for j in range(5)]+[Zsub[:,i]*Zsub[:,j] for i in range(5) for j in range(i+1,5)]
Xc=np.array(cols).T  # 16x16
coef,_,_,_=np.linalg.lstsq(Xc,evec,rcond=None)
print("exact fit residual",np.max(np.abs(Xc@coef-evec)))
# predict odd C=0
c0odd=[m for m in [format(i,'06b') for i in range(64)] if m[2]=='0' and sum(bits(m))%2==1]
Zo=np.array([[1 if m[i]=='1' else -1 for i in sub_facs] for m in c0odd])
Xo=np.array([np.ones(len(c0odd))]+[Zo[:,j] for j in range(5)]+[Zo[:,i]*Zo[:,j] for i in range(5) for j in range(i+1,5)]).T
ehat=Xo@coef
print("\nodd C=0 predictions (mains+2fi):")
for m,e in zip(c0odd,ehat):
    print("  ",m,f"e_hat={e:7.2f}  u_hat={738/(738+max(e,0)):.5f}")
# also mains-only
coef1=np.linalg.lstsq(Xc[:,:6],evec,rcond=None)[0]
ehat1=Xo[:,:6]@coef1
print("\nmains-only odd C=0 u_hat:", [round(738/(738+max(x,0)),4) for x in ehat1])
