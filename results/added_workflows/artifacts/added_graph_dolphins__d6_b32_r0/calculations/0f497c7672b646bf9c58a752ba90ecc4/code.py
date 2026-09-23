
import numpy as np, itertools
from wwb_helpers import fit_surface
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
allm=[''.join(map(str,b)) for b in itertools.product([0,1],repeat=6)]
allK=list(obs.keys()); allM=np.array([bits(m) for m in allK]); allY=np.array([obs[m] for m in allK])
normal=[i for i,m in enumerate(allK) if pop(m)>=2]
def basis(m,deg):
    x=bits(m); t=[1.0]+list(x)
    if deg>=2:
        for a in range(6):
            for b in range(a+1,6): t.append(x[a]*x[b])
    return np.array(t)
def ridgepred(tr,deg,alpha,xq):
    X=np.array([basis(allK[j],deg) for j in tr]); y=allY[tr]
    w=np.linalg.solve(X.T@X+alpha*np.eye(X.shape[1]),X.T@y)
    return basis(xq if isinstance(xq,str) else ''.join(map(str,xq.astype(int))),deg)@w
alphas=np.logspace(-5,3,33)
for deg in [1,2]:
    # per-cell loo base model preds
    bp=[]; 
    for i in normal:
        tr=[j for j in normal if j!=i]
        best=min(((np.mean([(ridgepred(tr,deg,a,allK[i])-allY[i])**2 for i in [i]]),a) for a in alphas))
        # fit alpha once by inner LOO
        def inner(a):
            e=0
            for k in tr:
                tr2=[j for j in tr if j!=k]
                e+=(ridgepred(tr2,deg,a,allK[k])-allY[k])**2
            return e
        a=min(alphas,key=inner)
        bp.append(ridgepred(tr,deg,a,allK[i]))
    bp=np.array(bp); m=allY[normal].mean()
    for lam in [0,0.25,0.5,0.75,1.0]:
        p=m+lam*(bp-m)
        print("deg%d blend lam=%.2f LOO rms %.5f"%(deg,lam,np.sqrt(np.mean((p-allY[normal])**2))))
# fit_surface format
pred=fit_surface(observations,degree=2)
print("fit_surface type",type(pred))
print("first:",pred[:3] if hasattr(pred,'__getitem__') else pred)
