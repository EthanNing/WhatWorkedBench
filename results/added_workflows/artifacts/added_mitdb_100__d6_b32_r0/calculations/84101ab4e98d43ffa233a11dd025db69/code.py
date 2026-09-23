
import numpy as np, json
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
# gate table (F,B) from observations
gate={}
for m,u in obs.items():
    A,B,C,D,E,F=bits(m)
    if (C,D,E)==(1,0,1): gate[(F,B)]=u
# h model for A=0,C=0 : main effects on (B,D,E,F) from 8 even A=0 C=0 points
A0C0=[m for m in obs if m[2]=='0' and m[0]=='0']
def z(b): return 1 if b=='1' else -1
B,D,E,F=[np.mean([z(m[i])* (738*(1-obs[m])/obs[m]) for m in A0C0]) for i in (1,3,4,5)]
mean_e=np.mean([738*(1-obs[m])/obs[m] for m in A0C0])
# main effect coef (on counts) via regression
Z=np.array([[1,z(m[1]),z(m[3]),z(m[4]),z(m[5])] for m in A0C0])
y=np.array([738*(1-obs[m])/obs[m] for m in A0C0])
coef=np.linalg.lstsq(Z,y,rcond=None)[0]
print("h model coeff [int,B,D,E,F]:",np.round(coef,3))
NF=738.0
pred={}
for i in range(64):
    m=format(i,'06b'); A,B,C,D,E,F=bits(m)
    if m in obs: pred[m]=obs[m]; continue
    if (C,D,E)==(1,0,1):
        v=gate[(F,B)]
    elif C==1:
        v=1.0
    else:
        if A==1: v=1.0
        else:
            e=coef[0]+coef[1]*z(m[1])+coef[2]*z(m[3])+coef[3]*z(m[4])+coef[4]*z(m[5])
            v=NF/(NF+max(e,0.0))
    pred[m]=min(1.0,max(0.0,v))
# verify reproduction of observed
err=max(abs(pred[m]-obs[m]) for m in obs if m in pred)
print("max reproduce error on observed:",err)
print("predicted (unobserved) values:")
for m in sorted(pred):
    if m not in obs: print("  ",m,round(pred[m],5))
json.dump({m:round(pred[m],6) for m in sorted(pred)}, open('/tmp/pred.json','w'))
print(json.dumps({m:round(pred[m],6) for m in sorted(pred)}))
