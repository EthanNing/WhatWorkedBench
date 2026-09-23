
import numpy as np
masks=[format(i,'06b') for i in range(64)]
X=np.array([[int(c) for c in m] for m in masks],float); Z1=2*X-1
cols=[np.ones(64)]+[Z1[:,j] for j in range(6)]+[Z1[:,a]*Z1[:,b] for a in range(6) for b in range(a+1,6)]
Z=np.column_stack(cols)
i0=masks.index('000000'); i63=masks.index('111111'); anchors=[i0,i63]
lam=1e-6; I22=np.eye(22)
edges=[(i,i^(1<<j)) for i in range(64) for j in range(6) if i<(i^(1<<j))]
def obj(S):
    M=Z[S]; s,ld=np.linalg.slogdet(M.T@M+lam*I22); return ld
def ecov(S):
    Ss=set(S); return sum(1 for i,k in edges if i in Ss and k in Ss)
np.random.seed(1)
rest0=[i for i in range(64) if i not in anchors]
# stage A multistart exchange for max obj
bestS=None; besto=-1e9
for t in range(40):
    S=anchors+list(np.random.choice(rest0,32,replace=False))
    rem=[i for i in rest0 if i not in S]
    imp=True
    while imp:
        imp=False; base=obj(S)
        for si in range(2,34):
            for oi in rem:
                S2=S.copy(); S2[si]=oi
                v=obj(S2)
                if v>base+1e-9:
                    rem.remove(oi); rem.append(S[si]); S=S2; base=v; imp=True; break
            if imp: break
    if obj(S)>besto: besto=obj(S); bestS=S.copy()
print("max obj",round(besto,4))
print("maxobj design chosen32",sorted(masks[i] for i in bestS if i not in anchors), "edges",ecov(bestS))
