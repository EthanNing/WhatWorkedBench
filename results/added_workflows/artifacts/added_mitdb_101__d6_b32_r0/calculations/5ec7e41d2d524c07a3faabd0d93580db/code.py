
import numpy as np, itertools
obs = {
"000000":0.9164420485175202,"111111":0.9686609686609686,
"000011":0.9405255878284924,"000101":0.3706896551724138,"000110":0.4305835010060362,
"001001":0.9811866859623734,"001010":0.5132075471698113,"001100":0.9897209985315712,
"001111":0.9456066945606695,"010001":0.9531914893617022,"010010":0.9110512129380054,
"010100":0.3891213389121339,"010111":0.3758099352051836,"011000":0.9941520467836257,
"011011":0.6686390532544378,"011101":0.9911504424778761,"011110":0.9405255878284924,
"100001":0.9855072463768116,"100010":0.9457579972183588,"100100":0.9912280701754386,
"100111":0.9897810218978103,"101000":0.9742120343839542,"101011":0.6544401544401545,
"101101":0.9882352941176471,"101110":0.9066666666666666,"110000":0.9869375907111756,
"110011":0.9883720930232558,"110101":0.9897510980966325,"110110":0.9883381924198251,
"111001":0.9955947136563876,"111010":0.6204379562043796,"111100":0.9926362297496318,
"111000":0.9941348973607038,"000111":0.42083333333333334,
}
allm=[ ''.join(b) for b in itertools.product('01',repeat=6)]
def bts(m): return tuple(int(c) for c in m)

def region(m):
    A,B,C,D,E,F=bts(m)
    if A==0 and C==0 and D==1: return 'LOW1'
    if C==1 and D==0 and E==1: return 'LOW2'
    if C==1 and D==1 and E==1: return 'MID'
    return 'HIGH'

from collections import defaultdict
groups=defaultdict(list)
for m,u in obs.items():
    groups[region(m)].append((m,u))
for g in ['LOW1','LOW2','MID','HIGH']:
    us=[u for _,u in groups[g]]
    print(f"{g:5s} n={len(us):2d} mean={np.mean(us):.4f} std={np.std(us):.4f} min={min(us):.4f} max={max(us):.4f}")
    print("   ", sorted([(m,round(u,4)) for m,u in groups[g]]))

# directly measured edges (Hamming-1 pairs among observed)
print("\n=== measured edges (Hamming-1 pairs) ===")
ks=list(obs)
for i in range(len(ks)):
    for j in range(i+1,len(ks)):
        a,b=ks[i],ks[j]
        if sum(x!=y for x,y in zip(a,b))==1:
            d=obs[a]-obs[b]
            idx=[k for k in range(6) if a[k]!=b[k]][0]
            print(f"{a} vs {b} flip {'ABCDEF'[idx]}: diff={d:+.4f}")
