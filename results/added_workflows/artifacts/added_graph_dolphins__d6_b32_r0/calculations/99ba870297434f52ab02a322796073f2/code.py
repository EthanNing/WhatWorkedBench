
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
allM=np.array([bits(m) for m in obs]); allY=np.array([obs[m] for m in obs]); allK=list(obs.keys())
normal=[i for i,m in enumerate(allK) if pop(m)>=2]
def fit_basis(idx,deg):
    X=[]; 
    for i in idx:
        x=allM[i]; t=[1.0]+list(x)
        if deg>=2:
            for a in range(6):
                for b in range(a+1,6): t.append(x[a]*x[b])
        if deg>=3:
            for a in range(6):
                for b in range(a+1,6):
                    for c in range(b+1,6): t.append(x[a]*x[b]*x[c])
        X.append(t)
    return np.array(X)
def pred_ridge(train,deg,alpha,xq):
    X=fit_basis(train,deg); y=allY[train]
    w=np.linalg.solve(X.T@X+alpha*np.eye(X.shape[1]),X.T@y)
    x=[]; t=[1.0]+list(xq)
    if deg>=2:
        for a in range(6):
            for b in range(a+1,6): t.append(xq[a]*xq[b])
    if deg>=3:
        for a in range(6):
            for b in range(a+1,6):
                for c in range(b+1,6): t.append(xq[a]*xq[b]*xq[c])
    return np.array(t)@w
def pred_knn(train,k,weighted,xq):
    d=np.abs(allM[train]-xq).sum(1)
    order=np.argsort(d)[:k]
    dd=d[order]; yy=allY[train][order]
    if weighted: w=1.0/(dd+1e-9)
    else: w=np.ones_like(dd,float)
    return (yy*w).sum()/w.sum()
def pred_gp(train,L,sig,xq):
    X=allM[train]; y=allY[train]
    D=np.abs(X[:,None,:]-X[None,:,:]).sum(2)
    K=np.exp(-D/L)+sig*sig*np.eye(len(train))
    dq=np.abs(X-xq).sum(1)
    kq=np.exp(-dq/L)
    alpha=np.linalg.solve(K,y)
    return kq@alpha
def loo(predfn,trainrule):
    errs=[]
    for i in normal:
        train=[j for j in trainrule if j!=i]
        errs.append(predfn(train,allM[i])-allY[i])
    return np.sqrt(np.mean(np.square(errs)))
TR_all=list(range(len(allK))); TR_norm=normal
print("constant(mean norm): %.5f"%loo(lambda tr,xq: allY[tr].mean() if False else np.mean([allY[j] for j in normal if True]),TR_norm))
for name,tr in [("norm",TR_norm),("all",TR_all)]:
    print("[train_%s] const %.5f | knn2 %.5f knn3 %.5f knn4 %.5f | knn2w %.5f knn3w %.5f | gp L=0.5 %.5f L=1 %.5f L=2 %.5f"%(
        name, loo(lambda t,x: allY[t].mean(),tr),
        loo(lambda t,x: pred_knn(t,2,False,x),tr), loo(lambda t,x: pred_knn(t,3,False,x),tr), loo(lambda t,x: pred_knn(t,4,False,x),tr),
        loo(lambda t,x: pred_knn(t,2,True,x),tr), loo(lambda t,x: pred_knn(t,3,True,x),tr),
        loo(lambda t,x: pred_gp(t,0.5,0.02,x),tr), loo(lambda t,x: pred_gp(t,1.0,0.02,x),tr), loo(lambda t,x: pred_gp(t,2.0,0.02,x),tr)))
    for deg in [1,2]:
        best=min((loo(lambda t,x: pred_ridge(t,deg,a,x),tr),a) for a in np.logspace(-4,2,13))
        print("   ridge d%d best %.5f alpha %.4g"%(deg,best[0],best[1]))
