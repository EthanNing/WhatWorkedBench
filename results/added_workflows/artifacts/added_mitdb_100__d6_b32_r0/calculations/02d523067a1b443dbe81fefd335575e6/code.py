
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

def col(t): return np.prod(Z[:,t],axis=1)
mains=[(i,) for i in range(6)]
pairs=[(i,j) for i in range(6) for j in range(i+1,6)]
triples=list(itertools.combinations(range(6),3))
X2=np.column_stack([np.ones(32)]+[col(t) for t in mains+pairs])
beta2,_,_,_=np.linalg.lstsq(X2,y,rcond=None)
resid=y-X2@beta2
# complementary pairs of triples
seen=set(); pairs3=[]
for t in triples:
    tc=tuple(sorted(set(range(6))-set(t)))
    if t not in seen:
        pairs3.append((t,tc)); seen.add(t); seen.add(tc)
print("10 complementary 3-factor pairs:")
G={}
for t,tc in pairs3:
    c=col(t)  # equals col(tc) on even fraction
    g=float((resid@c)/(c@c))
    G[(t,tc)]=g
    print("  ",''.join(names[i] for i in t),'~',''.join(names[i] for i in tc), f"  g={g:+.4f}")
print("\norder-2 model coefficients (main and pair):")
for nm,b in zip(['int']+[''.join(names[i] for i in t) for t in mains+pairs], beta2):
    print(f"   {nm:6s} {b:+.5f}")
