
import numpy as np, itertools, json

masks=[format(i,'06b') for i in range(64)]
X=np.array([[1 if c=='1' else -1 for c in m] for m in masks],float)  # 64x6

def modelmat(x,deg=2):
    n=x.shape[0]
    terms=[np.ones(n)]
    for i in range(6): terms.append(x[:,i])
    if deg>=2:
        for i in range(6):
            for j in range(i+1,6): terms.append(x[:,i]*x[:,j])
    if deg>=3:
        for i in range(6):
            for j in range(i+1,6):
                for k in range(j+1,6): terms.append(x[:,i]*x[:,j]*x[:,k])
    return np.column_stack(terms)

M2=modelmat(X,2)  # 64x22
anchors=['000000','111111']
aidx=[masks.index(a) for a in anchors]
candall=np.setdiff1d(np.arange(64),aidx)

def obj(chosen):
    obs=np.concatenate([chosen,aidx])
    U=np.setdiff1d(np.arange(64),obs)
    Xo=M2[obs]
    inv=np.linalg.inv(Xo.T@Xo+1e-9*np.eye(M2.shape[1]))
    lev=np.einsum('ij,jk,ik->i',M2[U],inv,M2[U])
    return lev.sum(), U

def exchange(k,trials=14,iters=40):
    best=None
    for t in range(trials):
        chosen=np.random.choice(candall,k,replace=False)
        cur,_=obj(chosen)
        for it in range(iters):
            improved=False
            for pos in np.random.permutation(k):
                cur_v=chosen[pos]; bv=cur; bc=cur_v
                cands=np.setdiff1d(candall,chosen)
                for c in cands:
                    ch=chosen.copy(); ch[pos]=c
                    v,_=obj(ch)
                    if v<bv-1e-12: bv=v; bc=c
                if bc!=cur_v:
                    chosen[pos]=bc; cur=bv; improved=True
            if not improved: break
        if best is None or cur<best[1]: best=(chosen.copy(),cur)
    return best

np.random.seed(0)
b24=exchange(24,trials=14)
b32=exchange(32,trials=8)
even=np.array([i for i in range(64) if masks[i].count('1')%2==0])
even_non=[i for i in even if i not in aidx]
oe,ue=obj(np.array(even_non))

def show(idx): return sorted(masks[i] for i in idx)
print("even-half variance sum:",round(oe,3),"U size",len(ue))
print("opt24 var:",round(b24[1],3),"opt32 var:",round(b32[1],3))
print("OPT24:",show(b24[0]))
print("OPT32:",show(b32[0]))
print("U24 size",len(obj(b24[0])[1]),"U32 size",len(obj(b32[0])[1]))
