
import numpy as np, itertools, collections
allm=[''.join(map(str,b)) for b in itertools.product([0,1],repeat=6)]
idx={m:i for i,m in enumerate(allm)}
def bv(m): return np.array([int(c) for c in m],float)
def bas2(x):
    t=[1.0]; t+=list(x)
    for i in range(6):
        for j in range(i+1,6): t.append(x[i]*x[j])
    return np.array(t)
B2=np.array([bas2(bv(m)) for m in allm])
def sub(vec): return np.array([1.0]+list(vec))
B1=np.array([sub(bv(m)) for m in allm])
def popAE(m): return sum(int(c) for c in m[:5])
known=[m for m in allm if popAE(m)<=1]
singles=['100000','010000','001000','000100','000010']
fixed=['000000','111111']+singles
cand=[m for m in allm if popAE(m)>=2 and m!='111111']
edges=[(m,m[:i]+'1'+m[i+1:]) for m in allm for i in range(6) if m[i]=='0']
def edge_mse(obs_set,B=B2,ridge=1e-6):
    X=np.array([B[idx[m]] for m in obs_set]); p=X.shape[1]
    M=X.T@X+ridge*np.eye(p); Minv=np.linalg.inv(M)
    zero=set(obs_set)|set(known); tot=0.0; exact=0
    for u,v in edges:
        U=[z for z in (u,v) if z not in zero]
        if not U: exact+=1; continue
        for a in U:
            for b in U: tot+=B[idx[a]]@Minv@B[idx[b]]
    return tot, exact
def search(sel, crit, passes=6):
    sel=list(sel)
    cur=crit(sel)[0]
    for _ in range(passes):
        improved=False
        for i in range(len(singles),len(sel)):
            s=sel[i]
            for c in cand:
                if c in sel: continue
                trial=sel[:]; trial[i]=c
                e=crit(trial)[0]
                if e<cur-1e-9:
                    sel=trial; cur=e; improved=True; break
            if improved: break
        if not improved: break
    return sel,cur
greedy=['000000','111111']+singles+['111110','111000','111011','111101','110010','110001','011111','011010','000111','100100','000110','010110','001100','010011','011001','101001','100011','010101','101111','011000','110100','010111','100110','011101','001111','110111','100010']
# note: rebuild greedy from previous run accurately:
greedy_sel=['000000','111111']+singles+['111110','111000','111011','111101','110010','110001','011111','011010','000111','100100','000110','010110','001100','010011','011001','101001','100011','010101','101111','011000','110100','010111','100110','011101','001111','110111','100010']
greedy_sel=[m for m in greedy_sel]
print(len(greedy_sel), len(set(greedy_sel)))
sug=['000000','111111']+singles+[m for m in ['000110','000111','001011','010010','010011','010100','010110','011001','011010','011100','100010','100111','101010','101101','101110','110000','110001','110010','110100','110101','110110','111011','111100']]
print("sug len",len(sug),len(set(sug)))
for name,sel in [("greedy",greedy_sel),("sug",sug)]:
    e,ex=edge_mse(sel); print(name,"edgeMSE %.3f exact %d"%(e,ex))
# Q5 face A=0
q5=['0'+m[1:] for m in allm]+['111111']
q5=list(dict.fromkeys(q5+known+singles))
print("q5 size",len(q5))
e,ex=edge_mse(sorted(set(q5))[:34]) if False else (None,None)
q5obs=[m for m in allm if m[0]=='0']+['100000','100001','111111']
print("q5obs",len(q5obs))
e,ex=edge_mse(q5obs); print("Q5 edgeMSE %.3f exact %d"%(e,ex))
# refine greedy via exchange
best_sel,best_e=search(greedy_sel, lambda s: edge_mse(s))
e,ex=edge_mse(best_sel); print("refined edgeMSE %.3f exact %d"%(e,ex))
print("refined purchase list:",sorted([m for m in best_sel if m not in ('000000','111111')]))
print("hist",collections.Counter(popAE(m) for m in best_sel))
# also under degree-1 truth
print("deg1 crit greedy: %.3f"%edge_mse(best_sel,B1)[0], " sug: %.3f"%edge_mse(sug,B1)[0], " Q5: %.3f"%edge_mse(q5obs,B1)[0])
