
import numpy as np, itertools
masks = [format(i,'06b') for i in range(64)]
X = np.array([[int(c) for c in m] for m in masks], dtype=float)  # 64x6
# +/-1 coding
Z1 = 2*X-1
# degree-2 design matrix
cols = [np.ones(64)]
for j in range(6): cols.append(Z1[:,j])
for a in range(6):
    for b in range(a+1,6):
        cols.append(Z1[:,a]*Z1[:,b])
Z = np.column_stack(cols)  # 64 x 22
i0 = masks.index('000000'); i63 = masks.index('111111')
anchors = [i0, i63]

def logdet(S):
    M = Z[S]
    G = M.T@M
    sign, ld = np.linalg.slogdet(G)
    return ld if sign>0 else -1e18

# greedy add
S = list(anchors)
remaining = [i for i in range(64) if i not in S]
while len(S) < 34:
    best=None; bestv=-1e18
    for i in remaining:
        v=logdet(S+[i])
        if v>bestv: bestv=v; best=i
    S.append(best); remaining.remove(best)
# exchange
improved=True
while improved:
    improved=False
    base=logdet(S)
    for si in range(len(S)):
        if S[si] in anchors: continue
        for oi in remaining:
            S2=S.copy(); S2[si]=oi
            v=logdet(S2)
            if v>base+1e-9:
                remaining.remove(oi); remaining.append(S[si])
                S=S2; base=v; improved=True
                break
        if improved: break
print("logdet", logdet(S))
print("chosen 32:", [masks[i] for i in S if i not in anchors])
print("all34:", sorted(masks[i] for i in S))
# edge coverage
Sset=set(S)
edges=[]
for i in range(64):
    for j in range(6):
        k=i^(1<<j)
        if i<k: edges.append((i,k))
cov=sum(1 for i,k in edges if i in Sset and k in Sset)
print("edges covered", cov, "of", len(edges))
