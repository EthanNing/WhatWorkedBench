
import numpy as np, itertools, json
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
"111000":0.9941348973607038,"000111":0.42083333333333334}
allm=[''.join(b) for b in itertools.product('01',repeat=6)]
def bts(m): return tuple(int(c) for c in m)
def region(m):
    A,B,C,D,E,F=bts(m)
    if A==0 and C==0 and D==1: return 'LOW1'
    if C==1 and D==0 and E==1: return 'LOW2'
    if C==1 and D==1 and E==1: return 'MID'
    return 'HIGH'
def xvec(m): return np.array([1.0 if c=='1' else -1.0 for c in m])
def terms_list(maxdeg):
    t=[()]
    for k in range(1,maxdeg+1):
        for S in itertools.combinations(range(6),k): t.append(S)
    return t
def design(masks,t):
    return np.column_stack([[np.prod([xvec(m)[j] for j in S]) for m in masks] for S in t])
hidx=[m for m in obs if region(m)=='HIGH']; yh=np.array([obs[m] for m in hidx])
def loo(deg,alpha):
    X=design(hidx,terms_list(deg)); p=X.shape[1]
    P=np.eye(p)*alpha; P[0,0]=0
    H=X@np.linalg.solve(X.T@X+P,X.T); e=yh-H@yh; h=np.diag(H)
    return np.sqrt(np.mean((e/np.clip(1-h,1e-9,None))**2))
grid=[(d,a) for d in [1,2,3] for a in [0.3,1,3,10,30,100,300,1000]]
scored=sorted((loo(d,a),d,a) for d,a in grid)
print("top HIGH models (rmse,deg,alpha):")
for s in scored[:6]: print("  ",round(s[0],5),s[1],s[2])
print("const:", round(np.sqrt(np.mean((yh-yh.mean())**2)),5))

# choose deg2 a=3 (robust) and also look at deg1 a=10
def pred_high(deg,alpha,test):
    X=design(hidx,terms_list(deg)); p=X.shape[1]
    P=np.eye(p)*alpha; P[0,0]=0
    beta=np.linalg.solve(X.T@X+P,X.T@yh)
    return design(test,terms_list(deg))@beta
hi_un=[m for m in allm if m not in obs and region(m)=='HIGH']
for (deg,alpha) in [(1,10),(2,3),(2,10),(3,300)]:
    pr=pred_high(deg,alpha,hi_un)
    print(f"\ndeg{deg} a{alpha}: min {pr.min():.4f} max {pr.max():.4f}")
    print("  ",{m:round(float(v),4) for m,v in zip(sorted(hi_un),pr[np.argsort(hi_un)])} if False else "")
    for m,v in zip(sorted(hi_un),pred_high(deg,alpha,sorted(hi_un))):
        print(f"   {m}:{v:.4f}",end="")
    print()
