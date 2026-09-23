import numpy as np, itertools
obs_list=[{"mask":"000000","utility":0.5},{"mask":"111111","utility":0.9019607843137254},
{"mask":"000111","utility":0.9019607843137255},{"mask":"011001","utility":0.8938869665513264},
{"mask":"101010","utility":0.8988850442137639},{"mask":"110100","utility":0.8942714340638216},
{"mask":"010010","utility":0.8592848904267589},{"mask":"001100","utility":0.8954248366013072},
{"mask":"100001","utility":0.8881199538638985},{"mask":"001011","utility":0.9015763168012303},
{"mask":"110101","utility":0.895040369088812},{"mask":"010011","utility":0.8938869665513264},
{"mask":"010100","utility":0.895040369088812},{"mask":"100111","utility":0.9011918492887352},
{"mask":"111011","utility":0.9011918492887351},{"mask":"100010","utility":0.8927335640138409},
{"mask":"110000","utility":0.8877354863514033},{"mask":"101101","utility":0.8954248366013072},
{"mask":"000101","utility":0.8961937716262975},{"mask":"111100","utility":0.895040369088812},
{"mask":"101110","utility":0.8988850442137639},{"mask":"110110","utility":0.8938869665513265},
{"mask":"011111","utility":0.9019607843137254},{"mask":"100100","utility":0.8946559015763168},
{"mask":"000110","utility":0.8750480584390619},{"mask":"011010","utility":0.9000384467512494},
{"mask":"101000","utility":0.8946559015763168},{"mask":"001001","utility":0.8954248366013072},
{"mask":"011100","utility":0.8958093041138024},{"mask":"110001","utility":0.8877354863514033},
{"mask":"000010","utility":0.8081507112648981},{"mask":"010000","utility":0.8836985774702037},
{"mask":"010110","utility":0.8842752787389466},{"mask":"110010","utility":0.8927335640138409}]
bvec=lambda m: np.array([int(c) for c in m],float)
sub=[o for o in obs_list if o['mask']!='000000']
mtr=[o['mask'] for o in sub]; ytr=np.array([o['utility'] for o in sub]); n=len(ytr)
Btr=np.array([bvec(m) for m in mtr])
def kernel_loo(W,theta):
    D=np.abs(Btr[:,None,:]-Btr[None,:,:]).sum(2)
    D=((Btr[:,None,:]-Btr[None,:,:])**2*W).sum(2)  # weighted sq dist
    out=np.zeros(n)
    for i in range(n):
        w=np.exp(-D[i]/(2*theta)); w[i]=0
        if w.sum()<1e-12: w[i]=1
        out[i]=(w*ytr).sum()/w.sum()
    return out
best=None
for wE in [1,4,9,25]:
 for wF in [1,4,9,25]:
  W=np.array([1,1,1,1,wE,wF],float)
  for th in [0.3,0.5,0.7,1.0,1.5,2.0,3.0]:
    p=kernel_loo(W,th); m=np.mean((p-ytr)**2)
    if best is None or m<best[0]: best=(m,wE,wF,th)
print('best kernel loo mse %.4g wE=%s wF=%s theta=%s'%best)
# stratified within (E,F) cell kNN over ABCD
def strat_knn(k,wtype):
    out=np.zeros(n)
    for i in range(n):
        same=(Btr[:,4]==Btr[i,4])&(Btr[:,5]==Btr[i,5]); same[i]=False
        d=((Btr[:,:4]-Btr[i,:4])**2).sum(1).astype(float)
        cand=np.where(same)[0]
        order=cand[np.argsort(d[cand])][:k]
        dd=d[order]
        w=np.ones(len(order)) if wtype=='u' else 1.0/(dd+0.5)
        out[i]=(w*ytr[order]).sum()/w.sum()
    return out
for k in (1,2,3,4,5):
    for wt in ('u','d'):
        p=strat_knn(k,wt); print('strat-knn k=%d %s mse %.4g rmse_k %.1f'%(k,wt,np.mean((p-ytr)**2),np.sqrt(np.mean((p-ytr)**2))*2601))
