
import numpy as np
masks=[format(i,'06b') for i in range(64)]
X=np.array([[int(c) for c in m] for m in masks],float); Z1=2*X-1
cols=[np.ones(64)]+[Z1[:,j] for j in range(6)]+[Z1[:,a]*Z1[:,b] for a in range(6) for b in range(a+1,6)]
Z=np.column_stack(cols)
i0=masks.index('000000'); i63=masks.index('111111'); anchors=[i0,i63]
lam=1e-6
edges=[(i,i^(1<<j)) for i in range(64) for j in range(6) if i<(i^(1<<j))]
def objS(S):
    M=Z[S]; G=M.T@M+lam*np.eye(22); s,ld=np.linalg.slogdet(G); return ld
def edges_cov(Sset):
    return sum(1 for i,k in edges if i in Sset and k in Sset)
def total(S,mu):
    Sset=set(S); return objS(S)+mu*edges_cov(Sset)
def optimize(mu):
    S=list(anchors); remaining=[i for i in range(64) if i not in S]
    for _ in range(32):
        best=None; bv=-np.inf
        for i in remaining:
            v=total(S+[i],mu)
            if v>bv: bv=v; best=i
        S.append(best); remaining.remove(best)
    imp=True
    while imp:
        imp=False; base=total(S,mu)
        for si in range(len(S)):
            if S[si] in anchors: continue
            for oi in remaining:
                S2=S.copy(); S2[si]=oi
                v=total(S2,mu)
                if v>base+1e-9:
                    remaining.remove(oi); remaining.append(S[si]); S=S2; base=v; imp=True; break
            if imp: break
    return S
for mu in [0.0,0.02,0.05,0.1,0.25]:
    S=optimize(mu); Sset=set(S)
    print("mu",mu,"obj",round(objS(S),3),"edges",edges_cov(Sset),"chosen",sorted(masks[i] for i in S if i not in anchors))
