
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
allmask=[format(i,'06b') for i in range(64)]
Mobs=list(obs.keys()); y=np.array([obs[m] for m in Mobs])
def zmat(ms): return 2*np.array([[int(c) for c in m] for m in ms],float)-1.0
Zo=zmat(Mobs)
I=lambda s: tuple(names.index(x) for x in s)
def col(Z,t): return np.prod(Z[:,list(t)],axis=1)
mains=[I(x) for x in names]
pairs=[I(a+b) for i,a in enumerate(names) for b in names[i+1:] if (a+b)!='AB']
terms=mains+pairs+[I('CDE'),I('CEF'),I('CDF'),I('DEF'),I('CDEF')]
X=np.column_stack([np.ones(len(Mobs))]+[col(Zo,t) for t in terms])
beta,_,_,_=np.linalg.lstsq(X,y,rcond=None)
Zall=zmat(allmask)
Xall=np.column_stack([np.ones(64)]+[col(Zall,t) for t in terms])
pred=Xall@beta
table={m:p for m,p in zip(allmask,pred)}
print("Full prediction table (observed marked *):")
for m in sorted(allmask):
    tag='*' if m in obs else ' '
    print(f"  {m}{tag} {table[m]:.5f}")
# odd gate points detail
print("\nodd gate points (C=1,D=0,E=1):", [m for m in allmask if m[2]=='1' and m[3]=='0' and m[4]=='1' and m not in obs])
for m in ['001011','011010','101010','111011']:
    print("  ",m, round(table[m],5))
