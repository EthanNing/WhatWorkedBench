
import numpy as np, random
masks=[format(i,'06b') for i in range(64)]
X=np.array([[int(c) for c in m] for m in masks],float); Z1=2*X-1
cols=[np.ones(64)]+[Z1[:,j] for j in range(6)]+[Z1[:,a]*Z1[:,b] for a in range(6) for b in range(a+1,6)]
Z=np.column_stack(cols)
i0=masks.index('000000'); i63=masks.index('111111'); anchors=[i0,i63]
lam=1e-6
edges=[(i,i^(1<<j)) for i in range(64) for j in range(6) if i<(i^(1<<j))]
def objS(S):
    M=Z[S]; G=M.T@M+lam*np.eye(22); s,ld=np.linalg.slogdet(G); return ld
def ecov(S):
    Ss=set(S); return sum(1 for i,k in edges if i in Ss and k in Ss)
random.seed(0)
best=None; bestscore=-1e9
for trial in range(60):
    rest=[i for i in range(64) if i not in anchors]
    S=anchors+random.sample(rest,32)
    o=objS(S)
    # local search: maximize edges s.t. obj>=thresh
    for it in range(20000):
        a=random.randrange(2,34); b=random.choice(rest)
        if S[a]==b: continue
        old=S[a]; S[a]=b
        o2=objS(S); e2=ecov(S)
        if o2>=77.05:
            o=o2; rest2=rest
            if random.random()<0.3 or True:
                pass
            # greedy acceptance
            e1=ecov(S)  # after move; recompute old edges hard -> use score
        # simpler: accept if improves edges and obj>=thresh
        # undo/redo cleanly
        # (we'll just do: accept if o2>=thresh and (e2>ecov_before))
        S[a]=b
        # compute score
        score=e2+ (0 if o2>=77.05 else -100)
        if score>=bestscore_trial if False else True:
            pass
    # fallback simple approach below
    # Instead: hill climb on edges with obj constraint
    o=objS(S)
    for it in range(20000):
        a=random.randrange(2,34); b=random.choice(rest)
        if S[a]==b: continue
        old=S[a]
        e_before=ecov(S)
        S[a]=b; o2=objS(S); e_after=ecov(S)
        if o2>=77.05 and e_after>=e_before:
            o=o2
        else:
            S[a]=old
    score=ecov(S)
    if score>bestscore:
        bestscore=score; best=S.copy()
print("best edges",bestscore,"obj",round(objS(best),3))
print("chosen32",sorted(masks[i] for i in best if i not in anchors))
print("all34",sorted(masks[i] for i in best))
