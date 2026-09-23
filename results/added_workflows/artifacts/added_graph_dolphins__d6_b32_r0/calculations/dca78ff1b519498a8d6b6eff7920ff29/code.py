
import numpy as np, itertools, json
obs=dict([("000000",0.5),("111111",0.74951171875),("000010",0.69482421875),("000100",0.759765625),
("000111",0.75830078125),("001000",0.759765625),("001011",0.75830078125),("001100",0.759765625),
("001101",0.759765625),("010000",0.75341796875),("010100",0.76416015625),("010101",0.75830078125),
("010110",0.73583984375),("010111",0.75830078125),("011000",0.75830078125),("011001",0.75830078125),
("011011",0.76123046875),("011110",0.75732421875),("011111",0.76123046875),("100000",0.755859375),
("100011",0.76123046875),("100101",0.751953125),("100110",0.76220703125),("101000",0.751953125),
("101010",0.76025390625),("101011",0.75439453125),("101101",0.7509765625),("101111",0.75439453125),
("110001",0.75634765625),("110110",0.76123046875),("111001",0.75634765625),("111010",0.76123046875),
("111100",0.75244140625),("111110",0.76220703125)])
singles=['100000','010000','001000','000100','000010']
obs['000001']=0.5
for s in singles: obs[s[:5]+str(1-int(s[5]))]=obs[s]
allm=[''.join(map(str,b)) for b in itertools.product([0,1],repeat=6)]
normal=[v for m,v in obs.items() if sum(int(c) for c in m[:5])>=2]
C=float(np.mean(normal))
print("n known exact:",len(obs)," normal n:",len(normal)," constant C=%.12f"%C)
pred={}
for m in allm:
    pred[m]=round(obs[m],12) if m in obs else round(C,12)
vals=np.array(list(pred.values()))
assert len(pred)==64 and np.isfinite(vals).all() and vals.min()>=0 and vals.max()<=1
# sanity: edge differences
ed=[]
for m in allm:
    for i in range(6):
        if m[i]=='0':
            mm=m[:i]+'1'+m[i+1:]; ed.append(pred[mm]-pred[m])
ed=np.array(ed); print("192 edge diffs: min %.4f max %.4f  n_zero %d"%(ed.min(),ed.max(),(ed==0).sum()))
print(json.dumps(pred))
