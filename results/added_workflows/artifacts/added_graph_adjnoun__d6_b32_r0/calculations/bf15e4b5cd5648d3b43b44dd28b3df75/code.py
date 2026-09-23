
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

Delta5 = meas["111111"]-meas["111110"]          # y(F=1)-y(F=0) at k=5 = -0.008028
def Delta(k):
    if k<=1: return 0.0
    return Delta5*(k-1)/4.0

def pk(m): return sum(int(c) for c in m[:5])
pred={}
for bits in itertools.product([0,1],repeat=6):
    m=''.join(map(str,bits)); k=pk(m)
    if m in meas:
        pred[m]=float(meas[m]); continue
    Fp=pk.__self__ if False else None
    p_mask=m[:5]+str(sum(bits[:5])%2)   # the measured principal partner (same A-E)
    Fp=int(p_mask[5]); f=bits[5]
    h=(f-Fp)*Delta(k)
    pred[m]=meas[p_mask]+h
    # consistency: this must reproduce the two measured complement points
    if m in ("000001","111110"):
        print("check",m,"pred",pred[m],"true",meas[m])

vals=np.array(list(pred.values()))
print("range %.4f %.4f"%(vals.min(),vals.max()))
print("argmax:",max(pred,key=pred.get),round(pred[max(pred,key=pred.get)],5))
# edge stats
edges=[]
for b in itertools.product([0,1],repeat=6):
    m=''.join(map(str,b))
    for i in range(6):
        bb=list(b); bb[i]^=1; m2=''.join(map(str,bb))
        edges.append(abs(pred[m]-pred[m2]))
edges=np.array(edges); print("edge mean %.4f max %.4f  >0.05:%d >0.1:%d"%(edges.mean(),edges.max(),(edges>0.05).sum(),(edges>0.1).sum()))
print(json.dumps(pred))
