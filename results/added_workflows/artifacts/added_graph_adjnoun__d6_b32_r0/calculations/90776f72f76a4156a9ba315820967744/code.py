
import numpy as np, itertools, json
meas = {"000000":0.5,"000011":0.7433217993079584,"000101":0.6401384083044982,
"000110":0.7444290657439446,"001001":0.6417301038062283,"001010":0.7417993079584775,
"001100":0.6417301038062283,"001111":0.740553633217993,"010001":0.6279584775086505,
"010010":0.7439446366782007,"010100":0.6365397923875432,"010111":0.7333564013840831,
"011000":0.6419377162629758,"011011":0.7329411764705882,"011101":0.6550865051903113,
"011110":0.7415224913494809,"100001":0.6416608996539792,"100010":0.7469204152249136,
"100100":0.6414532871972318,"100111":0.7382006920415225,"101000":0.6414532871972318,
"101011":0.7398615916955018,"101101":0.6411764705882352,"101110":0.7423529411764707,
"110000":0.6460899653979237,"110011":0.7435294117647059,"110101":0.6693425605536332,
"110110":0.7444290657439447,"111001":0.6690657439446366,"111010":0.742076124567474,
"111100":0.6422145328719723,"111111":0.7340484429065743,"000001":0.5,"111110":0.742076124567474}

# delta(k) = y(F=1) - y(F=0), anchored at k=0 -> 0, k=1 -> 0 (single feature ranking invariant),
# k=5 -> measured difference (111110 vs 111111) = -0.008028
delta5 = meas["111110"]-meas["111111"]
def delta(k):
    if k<=1: return 0.0
    return delta5*(k-1)/4.0
print("delta5 =",delta5)

def parity5(m): return sum(int(c) for c in m[:5])%2

pred={}
for bits in itertools.product([0,1],repeat=6):
    m=''.join(map(str,bits))
    k=sum(bits[:5])
    if m in meas:
        pred[m]=float(meas[m]); continue
    # complement vertex: same A-E, F flipped -> principal counterpart
    pm=m[:5]+str(1-bits[5])
    assert pm in meas, (m,pm)
    f=bits[5]; fp=1-f
    h=(f-fp)*delta(k)
    pred[m]=meas[pm]+h

# clip and report
vals=np.array([pred[''.join(map(str,b))] for b in itertools.product([0,1],repeat=6)])
print("range",vals.min(),vals.max(),"n",len(vals))
print("argmax mask:",max(pred,key=pred.get),pred[max(pred,key=pred.get)])
# sanity: reproduce measured exactly
print("max meas err:",max(abs(pred[m]-meas[m]) for m in meas))
# edge-difference sanity: distribution
import itertools
def idx(m): return int(m,2)
edges=[]
for b in itertools.product([0,1],repeat=6):
    m=''.join(map(str,b))
    for i in range(6):
        bb=list(b); bb[i]^=1; m2=''.join(map(str,bb))
        edges.append(abs(pred[m]-pred[m2]))
edges=np.array(edges)
print("edge diffs: mean %.4f max %.4f  #>0.05: %d #>0.1: %d"%(edges.mean(),edges.max(),sum(edges>0.05),sum(edges>0.1)))
print(json.dumps(pred))
