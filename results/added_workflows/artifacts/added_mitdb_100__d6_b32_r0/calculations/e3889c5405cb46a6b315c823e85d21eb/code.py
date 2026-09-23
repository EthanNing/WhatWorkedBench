
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
Z=2*np.array([[int(c) for c in m] for m in masks])-1.0
I=lambda s: tuple(names.index(x) for x in s)
def col(t): return np.prod(Z[:,list(t)],axis=1)

# mechanistic term set: main, 2fi except AB replaced by CDEF, plus CDE,CEF,CDF,DEF
mains=[I(x) for x in names]
pairs=[I(a+b) for i,a in enumerate(names) for b in names[i+1:]]
pairs=[p for p in pairs if p!=I('AB')]
terms=mains+pairs+[I('CDE'),I('CEF'),I('CDF'),I('DEF'),I('CDEF')]
X=np.column_stack([np.ones(32)]+[col(t) for t in terms])
beta,_,_,_=np.linalg.lstsq(X,y,rcond=None)
resid=y-X@beta
print("nparams",len(terms)+1,"maxabs_resid",round(np.max(np.abs(resid)),5),"rmse",round(np.sqrt(np.mean(resid**2)),5))
for t,b in zip(['int']+[''.join(names[i] for i in t) for t in terms],beta):
    if abs(b)>0.0008: print(f"   {t:6s} {b:+.5f}")
print("residuals sorted:")
for i in np.argsort(-np.abs(resid))[:12]:
    print("  ",masks[i],round(resid[i],5))
