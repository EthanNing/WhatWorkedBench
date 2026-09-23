
import numpy as np, itertools, json
obs = [("000000",0.5),("000011",0.7433217993079584),("000101",0.6401384083044982),
("000110",0.7444290657439446),("001001",0.6417301038062283),("001010",0.7417993079584775),
("001100",0.6417301038062283),("001111",0.740553633217993),("010001",0.6279584775086505),
("010010",0.7439446366782007),("010100",0.6365397923875432),("010111",0.7333564013840831),
("011000",0.6419377162629758),("011011",0.7329411764705882),("011101",0.6550865051903113),
("011110",0.7415224913494809),("100001",0.6416608996539792),("100010",0.7469204152249136),
("100100",0.6414532871972318),("100111",0.7382006920415225),("101000",0.6414532871972318),
("101011",0.7398615916955018),("101101",0.6411764705882352),("101110",0.7423529411764707),
("110000",0.6460899653979237),("110011",0.7435294117647059),("110101",0.6693425605536332),
("110110",0.7444290657439447),("111001",0.6690657439446366),("111010",0.742076124567474),
("111100",0.6422145328719723),("111111",0.7340484429065743),("000001",0.5),("111110",0.742076124567474)]

names=["A","B","C","D","E","F"]
def bits(m): return [int(c) for c in m]
def design(masks, degree):
    cols=[np.ones(len(masks))]
    B=np.array([bits(m) for m in masks],dtype=float)
    S=2*B-1  # +-1 coding
    for i in range(6): cols.append(S[:,i])
    if degree>=2:
        for i in range(6):
            for j in range(i+1,6): cols.append(S[:,i]*S[:,j])
    return np.column_stack(cols)

masks=[o[0] for o in obs]; y=np.array([o[1] for o in obs])
X2=design(masks,2)
coef,res,rank,sv=np.linalg.lstsq(X2,y,rcond=None)
pred=X2@coef
r=y-pred
print("degree2 OLS rank",rank,"resid rms %.3e max %.3e"%(np.sqrt((r**2).mean()),np.abs(r).max()))
labs=["1"]+names+[names[i]+names[j] for i in range(6) for j in range(i+1,6)]
for l,c in sorted(zip(labs,coef),key=lambda t:-abs(t[1])): print(f"  {l:4s} {c:+.5f}")

# degree1
X1=design(masks,1)
c1,_,_,_=np.linalg.lstsq(X1,y,rcond=None)
r1=y-X1@c1
print("degree1 rms %.4e max %.4e"%(np.sqrt((r1**2).mean()),np.abs(r1).max()))
print("main effects deg1:",dict(zip(["1"]+names,np.round(c1,5))))

# fit on the 32 principal only, predict the 2 extras
prim=[o[0] for o in obs if o[0] not in ("000001","111110")]
yp=np.array([o[1] for o in obs if o[0] not in ("000001","111110")])
cp,_,_,_=np.linalg.lstsq(design(prim,2),yp,rcond=None)
for m,true in [("000001",0.5),("111110",0.742076124567474)]:
    pr=design([m],2)@cp
    print("extrap",m,"pred %.5f true %.5f"%(pr[0],true))
