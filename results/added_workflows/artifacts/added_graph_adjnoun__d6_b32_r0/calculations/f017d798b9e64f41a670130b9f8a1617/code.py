
import numpy as np, itertools
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
def bits(m): return np.array([int(c) for c in m])
names="ABCDEF"
rows=[]
for m,y in obs:
    b=bits(m); d=dict(zip(names,b))
    E=d['E']; v=int(any(d[k] for k in "ABCD")); F=d['F']; r=sum(d[k] for k in "ABCD")
    rows.append((m,y,b,d,E,v,F,r))
# group means by (E,v)
for E in (0,1):
    for v in (0,1):
        g=[t for t in rows if t[4]==E and t[5]==v]
        ys=[t[1] for t in g]
        print(f"E={E} v={v} n={len(g)} mean={np.mean(ys):.4f} sd={np.std(ys,ddof=1) if len(g)>1 else 0:.4f} min={min(ys):.4f} max={max(ys):.4f}")
        if E==0 and v==1:
            for t in sorted(g,key=lambda z:-z[1]): print("    ",t[0],f"{t[1]:.4f}","r=",t[7],"F=",t[2][5])
print()
# residuals around model  y ~ 1,E,v,E*v
X=np.column_stack([np.ones(len(rows)),[t[4] for t in rows],[t[5] for t in rows],[t[4]*t[5] for t in rows]])
y=np.array([t[1] for t in rows])
c,_,_,_=np.linalg.lstsq(X,y,rcond=None)
res=y-X@c
print("model 1+E+v+Ev coefs:",np.round(c,5),"rms %.4e max %.4e"%(np.sqrt((res**2).mean()),np.abs(res).max()))
order=np.argsort(-np.abs(res))
for i in order[:12]: print("   resid",rows[i][0],f"{res[i]:+.4f}","y=",f"{y[i]:.4f}","v=",rows[i][5],"E=",rows[i][4],"r=",rows[i][7],"F=",rows[i][2][5])
