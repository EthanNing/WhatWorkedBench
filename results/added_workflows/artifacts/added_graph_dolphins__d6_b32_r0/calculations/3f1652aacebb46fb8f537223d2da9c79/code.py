
import numpy as np, itertools
from wwb_helpers import suggest_experiments

# --- structural analysis of the workflow ---
# mask bits: A B C D E F
# Key facts from source:
#  * if none of A-E enabled -> features=zeros -> LR constant -> AUC 0.5
#  * if exactly one of A-E enabled -> single feature; per-column monotone scaling (F) does not change ranking
#    -> AUC independent of F.
def bv(m): return np.array([int(c) for c in m], float)
LOW = []
# S = active set among A..E (indices 0..4), F index 5
for sos in itertools.product([0,1],repeat=5):
    for f in [0,1]:
        m = ''.join(map(str,sos+(f,)))
        if sum(sos)<=1: LOW.append(m)
print("low cells (known):", LOW, len(LOW))

def basis(x):  # degree2
    t=[1.0]
    t+=list(x)
    for i in range(6):
        for j in range(i+1,6):
            t.append(x[i]*x[j])
    return np.array(t)
P=22
allm=[''.join(map(str,b)) for b in itertools.product([0,1],repeat=6)]
Xall=np.array([basis(bv(m)) for m in allm])   # 64x22
idx={m:i for i,m in enumerate(allm)}

# I-optimal design: fixed points + forward select 27 from |S|>=2 (excl 111111)
fixed=['000000','100000','010000','001000','000100','000010','111111']
cand=[m for m in allm if sum(int(c) for c in m[:5])>=2 and m!='111111']
print("n candidates:",len(cand))

def crit(sel, w=None, ridge=1e-6):
    X=np.array([Xall[idx[m]] for m in sel])
    M=X.T@X + ridge*np.eye(P)
    Minv=np.linalg.inv(M)
    v=np.einsum('ij,jk,ik->i',Xall,Minv,Xall)
    return v.sum(), v

sel=list(fixed)
for it in range(27):
    best=None
    for c in cand:
        if c in sel: continue
        s,_=crit(sel+[c])
        if best is None or s<best[0]: best=(s,c)
    sel.append(best[1])
print("I-optimal selection (27):")
print(sorted(sel))
s,v=crit(sel)
print("I-crit:",s, "max pred var:",v.max())
# stats
import collections
print("popcount hist:", collections.Counter(sum(int(c) for c in m[:5]) for m in sel))
print("F on count:", sum(1 for m in sel if m[5]=='1'))
# compare to suggest_experiments list
sug=suggest_experiments(observations, count=32, degree=2)
print("suggest list:", sorted(sug))
print("suggest popcount hist:", collections.Counter(sum(int(c) for c in m[:5]) for m in sug))
