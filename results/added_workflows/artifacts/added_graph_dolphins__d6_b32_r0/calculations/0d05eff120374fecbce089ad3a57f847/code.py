
import numpy as np, itertools
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
def bits(m): return np.array([int(c) for c in m],float)
def pop(m): return sum(int(c) for c in m[:5])
allK=list(obs.keys()); allM=np.array([bits(m) for m in allK]); allY=np.array([obs[m] for m in allK])
normal=[i for i,m in enumerate(allK) if pop(m)>=2]
def flip(m,d): return m[:d]+str(1-int(m[d]))+m[d+1:]
def loo2(pred):
    errs=[pred([j for j in normal if j!=i], allM[i], allK[i])-allY[i] for i in normal]
    return np.sqrt(np.mean(np.square(errs)))
def knn(tr,xq,k):
    d=np.abs(allM[tr]-xq).sum(1); o=np.argsort(d)[:k]; return allY[tr][o].mean()
def const(tr,xq,m): return allY[tr].mean()
def fcount(tr,xq,m,alpha):
    cm=flip(m,5); base=allY[tr].mean()
    if cm in obs and allK.index(cm) in tr: return alpha*allY[allK.index(cm)]+(1-alpha)*base
    return base
def fknn(tr,xq,m,w):
    cm=flip(m,5); base=knn(tr,xq,4)
    if cm in obs and allK.index(cm) in tr: return w*allY[allK.index(cm)]+(1-w)*base
    return base
print("const  %.5f"%loo2(const))
print("knn4   %.5f"%loo2(lambda t,x,m: knn(t,x,4)))
print("knn6   %.5f"%loo2(lambda t,x,m: knn(t,x,6)))
for a in [0.25,0.5,0.75,1.0]: print("fcount a=%.2f %.5f"%(a,loo2(lambda t,x,m,a=a: fcount(t,x,m,a))))
for w in [0.3,0.5,0.7]: print("fknn w=%.2f %.5f"%(w,loo2(lambda t,x,m,w=w: fknn(t,x,m,w))))
hs=[obs[flip(m,5)]-obs[m] for m in allK if pop(m)>=2 and flip(m,5) in obs and pop(flip(m,5))>=2]
print("Fpair diffs normal:",sorted(round(h,5) for h in hs),"rms %.5f"%np.sqrt(np.mean(np.square(hs))))
allm=[''.join(map(str,b)) for b in itertools.product([0,1],repeat=6)]
unobs=[m for m in allm if m not in obs]
print("unobs n=%d withcp:%d"%(len(unobs),sum(1 for m in unobs if flip(m,5) in obs)))
print(unobs)
Y=allY[normal]; P=np.array([pop(allK[i]) for i in normal]); E=np.array([int(allK[i][4]) for i in normal]); F=np.array([int(allK[i][5]) for i in normal])
for nm,v in [("pop",P),("E",E),("F",F)]: print(nm,"corr %.3f"%np.corrcoef(v,Y)[0,1])
