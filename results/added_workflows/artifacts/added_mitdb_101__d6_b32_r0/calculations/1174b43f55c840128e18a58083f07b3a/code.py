
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
T2=terms_list(2)
hidx=[m for m in obs if region(m)=='HIGH']; yh=np.array([obs[m] for m in hidx])
X=design(hidx,T2); p=X.shape[1]; a=3.0
P=np.eye(p)*a; P[0,0]=0
beta=np.linalg.solve(X.T@X+P,X.T@yh)
means={g:np.mean([obs[m] for m in obs if region(m)==g]) for g in ['LOW1','LOW2','MID']}
hi_un=sorted([m for m in allm if m not in obs and region(m)=='HIGH'])
hi_pred=design(hi_un,T2)@beta
pred={}
for m in allm:
    if m in obs: pred[m]=obs[m]
    elif region(m)=='HIGH': pred[m]=float(hi_pred[hi_un.index(m)])
    else: pred[m]=float(means[region(m)])
pred={m:float(round(min(1.0,max(0.0,v)),6)) for m,v in pred.items()}
s=json.dumps(pred)
print(s)
print("n=",len(pred))
