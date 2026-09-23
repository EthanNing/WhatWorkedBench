
import numpy as np
masks=[format(i,'06b') for i in range(64)]
X=np.array([[int(c) for c in m] for m in masks],float); Z1=2*X-1
cols=[np.ones(64)]+[Z1[:,j] for j in range(6)]+[Z1[:,a]*Z1[:,b] for a in range(6) for b in range(a+1,6)]
Z=np.column_stack(cols)
i0=masks.index('000000'); i63=masks.index('111111'); anchors=[i0,i63]
lam=1e-6
def obj(S):
    M=Z[S]; G=M.T@M+lam*np.eye(22); sign,ld=np.linalg.slogdet(G); return ld
S=list(anchors); remaining=[i for i in range(64) if i not in S]
while len(S)<34:
    best=None; bestv=-np.inf
    for i in remaining:
        v=obj(S+[i])
        if v>bestv: bestv=v; best=i
    S.append(best); remaining.remove(best)
improved=True
while improved:
    improved=False; base=obj(S)
    for si in range(len(S)):
        if S[si] in anchors: continue
        for oi in remaining:
            S2=S.copy(); S2[si]=oi
            v=obj(S2)
            if v>base+1e-7:
                remaining.remove(oi); remaining.append(S[si]); S=S2; base=v; improved=True; break
        if improved: break
print("obj",round(obj(S),3))
chosen=[masks[i] for i in S if i not in anchors]
print("chosen32:", sorted(chosen))
Sset=set(S)
edges=[(i,i^(1<<j)) for i in range(64) for j in range(6) if i<(i^(1<<j))]
cov=sum(1 for i,k in edges if i in Sset and k in Sset)
print("edges covered",cov,"of",len(edges))
# also coverage profile per factor
for j in range(6):
    e=[(i,i^(1<<j)) for i in range(64) if i<(i^(1<<j))]
    print("factor",j,"cov",sum(1 for i,k in e if i in Sset and k in Sset),"/",len(e))
