
import numpy as np
from itertools import combinations
obs=[("000000",0.9541284403669725),("111111",1.0),("000001",0.9771812080536912),("000010",0.9667994687915007),("000011",0.9837837837837838),("000100",0.9567567567567568),("000110",0.7906976744186046),("001000",0.9688385269121813),("001010",0.5536121673003802),("001101",0.8426073131955485),("010000",0.9732620320855615),("010111",0.7805695142378559),("011001",1.0),("011011",0.6953199617956065),("011100",0.9972451790633609),("011101",0.9972451790633609),("011110",0.8845686512758202),("100000",0.9667994687915007),("100101",1.0),("100111",0.8481012658227848),("101000",0.9688385269121813),("101001",0.9688385269121813),("101010",0.5536121673003802),("101011",0.6854990583804144),("101100",0.8426073131955485),("101101",0.8426073131955485),("101110",0.9904761904761905),("110000",0.9837837837837838),("110001",1.0),("110011",1.0),("110100",1.0),("110110",0.8389154704944178),("111000",1.0),("111010",0.6195744680851064)]
masks=[format(i,'06b') for i in range(64)]; midx={m:i for i,m in enumerate(masks)}
X=np.array([[int(c) for c in m] for m in masks],float); Z1=2*X-1
obs_m=[midx[m] for m,u in obs]; y=np.array([u for m,u in obs]); n=len(y)
def design(deg,Ms=None):
    cols=[np.ones(len(Ms))]
    for d in range(1,deg+1):
        for comb in combinations(range(6),d):
            col=np.ones(len(Ms))
            for j in comb: col*=Ms[:,j]
            cols.append(col)
    return np.column_stack(cols)
def loo_ridge(deg,alphas):
    Z=design(deg,Z1)[obs_m]; A=Z.T@Z; 
    best=(9,None)
    for al in alphas:
        B=np.linalg.solve(A+al*np.eye(A.shape[0]),Z.T)
        H=Z@B  # hat
        # loo residual via h_ii
        hii=np.diag(H); res=(y-Z@(B@y))/(1-hii)
        rm=np.sqrt(np.mean(res**2))
        if rm<best[0]: best=(rm,al)
    return best
for deg in [1,2,3]:
    al=np.logspace(-6,2,40)
    print("ridge deg",deg,loo_ridge(deg,al))
# GP hamming kernel
def gp_loo(theta, noise):
    d=np.array([[np.sum(masks[obs_m[i]]!=masks[obs_m[j]]) for j in range(n)] for i in range(n)],float)
    K=np.exp(-theta*d)+noise*np.eye(n)
    Kinv=np.linalg.inv(K)
    loo=np.zeros(n)
    for i in range(n):
        idx=[j for j in range(n) if j!=i]
        k=K[np.ix_(idx,idx)]; ki=np.linalg.inv(k)
        loo[i]=K[i,idx]@ki@y[idx]
    return np.sqrt(np.mean((loo-y)**2))
best=(9,)
for theta in [0.1,0.2,0.3,0.5,0.8,1.2,2.0]:
    for noise in [1e-4,1e-3,1e-2,5e-2]:
        rm=gp_loo(theta,noise)
        if rm<best[0]: best=(rm,theta,noise)
print("GP best",best)
# nearest neighbor
pred=np.zeros(n)
for i in range(n):
    d=[(np.sum(masks[obs_m[i]]!=masks[obs_m[j]]),j) for j in range(n) if j!=i]
    d.sort()
    # weighted avg of k nearest with weights 1/d^2
    ws=[1.0/(dd*dd) for dd,j in d[:5]]
    vs=[y[j] for dd,j in d[:5]]
    pred[i]=np.sum(np.array(ws)*np.array(vs))/np.sum(ws)
print("5NN loo rmse",np.sqrt(np.mean((pred-y)**2)))
