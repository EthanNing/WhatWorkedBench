
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
start=['000001','000011','000100','000110','001000','001010','001101','001111','010000','010010','010101','010111','011001','011011','011100','011110','100000','100010','100101','100111','101001','101010','101011','101100','101110','110001','110011','110100','110110','111000','111010','111101']
S0=anchors+[masks.index(m) for m in start]
def hill(thresh, S, iters=400):
    S=S.copy(); rest=[i for i in range(64) if i not in S]
    for _ in range(iters):
        bi=None; be=ecov(S); bo=obj(S)
        for si in range(2,34):
            for oi in rest:
                S2=S.copy(); S2[si]=oi
                o2=obj(S2)
                if o2<thresh: continue
                e2=ecov(S2)
                if e2>be or (e2==be and o2>bo+1e-9):
                    be=e2; bo=o2; bi=(si,oi)
        if bi is None: break
        si,oi=bi; rest.remove(oi); rest.append(S[si]); S[si]=oi
    return S
for th in [77.29,77.2,77.0,76.5,75.5,74.0,72.0]:
    S=hill(th,S0)
    print("thresh",th,"obj",round(obj(S),3),"edges",ecov(S))
    print("  chosen",sorted(masks[i] for i in S if i not in anchors))
