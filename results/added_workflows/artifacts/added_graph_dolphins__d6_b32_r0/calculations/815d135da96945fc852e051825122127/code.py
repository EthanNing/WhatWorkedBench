
import numpy as np, itertools, collections
allm=[''.join(map(str,b)) for b in itertools.product([0,1],repeat=6)]
idx={m:i for i,m in enumerate(allm)}
def bv(m): return np.array([int(c) for c in m],float)
def basis(x):
    t=[1.0]; t+=list(x)
    for i in range(6):
        for j in range(i+1,6): t.append(x[i]*x[j])
    return np.array(t)
B=np.array([basis(bv(m)) for m in allm])  # 64x22
def popAE(m): return sum(int(c) for c in m[:5])

# known-exactly set (no measurement needed): popcount(AE)<=1
known=[m for m in allm if popAE(m)<=1]
print("known exactly (12):",sorted(known),len(known))
# fixed observed points
singles=['100000','010000','001000','000100','000010']
fixed=['000000','111111']+singles
cand=[m for m in allm if popAE(m)>=2 and m!='111111']
edges=[(m,i) for m in allm for i in range(6) if m[i]=='0' for _ in [0]]
edges=[(m, m[:i]+'1'+m[i+1:]) for m in allm for i in range(6) if m[i]=='0']
print("n edges:",len(edges))

def edge_mse(obs_set):
    X=np.array([B[idx[m]] for m in obs_set])
    M=X.T@X+1e-6*np.eye(22); Minv=np.linalg.inv(M)
    zero=set(obs_set)|set(known)
    tot=0.0; exact=0
    for u,v in edges:
        U=[z for z in (u,v) if z not in zero]
        if not U: exact+=1; continue
        s=1.0
        val=0.0
        for a in U:
            for b in U:
                val+= s*B[idx[a]]@Minv@B[idx[b]]
        tot+=val
    return tot, exact

print("baseline I-opt design edgeMSE:", edge_mse(sorted(sel)) if 'sel' in dir() else None)

# greedy forward selection minimizing expected edge MSE
sel=list(fixed)
for it in range(27):
    best=None
    for c in cand:
        if c in sel: continue
        e,ex=edge_mse(sel+[c])
        if best is None or e<best[0]: best=(e,c,ex)
    sel.append(best[1])
    if it%9==0: print("iter",it,"added",best[1],"edgeMSE %.4f"%best[0],"exact",best[2])
e,ex=edge_mse(sel)
print("FINAL edgeMSE %.4f"%e,"exact edges",ex,"/192")
print("purchase list (32):", sorted([m for m in sel if m not in ('000000','111111')]))
print("popcount hist:",collections.Counter(popAE(m) for m in sel),"F on:",sum(m[5]=='1' for m in sel))
# reference: suggest list
sug=['000010','000101','000110','000111','001001','001011','001100','010000','010010','010011','010100','010110','011001','011010','011100','011111','100001','100010','100100','100111','101000','101010','101101','101110','110000','110001','110010','110100','110101','110110','111011','111100']
e2,ex2=edge_mse(sug); print("suggest-list edgeMSE %.4f exact"%e2,ex2)
