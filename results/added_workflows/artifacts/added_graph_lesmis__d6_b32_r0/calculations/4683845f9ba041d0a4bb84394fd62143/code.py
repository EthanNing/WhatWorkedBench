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
def fullp(X,deg):
    cols=[]
    for dd in range(1,deg+1):
        for c in itertools.combinations(range(6),dd): cols.append(np.prod(X[:,c],axis=1))
    return np.column_stack(cols)
def mk_e(X): return np.column_stack([X[:,i] for i in range(6)]+[X[:,4]*X[:,j] for j in (0,1,2,3,5)])
def mk_ef(X): return np.column_stack([X[:,i] for i in range(6)]+[X[:,4]*X[:,j] for j in (0,1,2,3,5)]+[X[:,5]*X[:,j] for j in (0,1,2,3)])
def mk_s(X):
    e=X[:,4];f=X[:,5];cols=[]
    for ev in (0,1):
        for fv in (0,1):
            ind=((e==ev)&(f==fv)).astype(float); cols+=[ind,ind*X[:,0],ind*X[:,1],ind*X[:,2],ind*X[:,3]]
    return np.column_stack(cols)
def loo_pred(Xf,y,alpha):
    out=np.zeros(len(y))
    for i in range(len(y)):
        tr=np.ones(len(y),bool); tr[i]=False
        Xt=Xf[tr]; yt=y[tr]; mx=Xt.mean(0); my=yt.mean()
        w=np.linalg.solve((Xt-mx).T@(Xt-mx)+alpha*np.eye(Xt.shape[1]),(Xt-mx).T@(yt-my))
        out[i]=(Xf[i]-mx)@w+my
    return out
def loo_knn(X,y,k):
    out=np.zeros(len(y))
    D=(X[:,None,:]!=X[None,:,:]).sum(2).astype(float)
    for i in range(len(y)):
        d=D[i].copy(); d[i]=1e9
        nn=np.argsort(d)[:k]; w=1.0/(d[nn]+0.0); w=w/w.sum()
        out[i]=(w*y[nn]).sum()
    return out
L={}
L['deg2']=loo_pred(fullp(Btr,2),ytr,10)
L['knn3']=loo_knn(Btr,ytr,3)
L['knn4']=loo_knn(Btr,ytr,4)
L['strat']=loo_pred(mk_s(Btr),ytr,1.0)
L['efint']=loo_pred(mk_ef(Btr),ytr,1.0)
for nm in L: print(nm,'LOO mse %.4g  rmse(k) %.1f'%(np.mean((L[nm]-ytr)**2), np.sqrt(np.mean((L[nm]-ytr)**2))*2601))
combos={'deg2+efint':['deg2','efint'],'all4':['deg2','knn3','strat','efint'],'knn3+strat':['knn3','strat'],'deg2+knn3+strat':['deg2','knn3','strat'],'knn3+knn4+strat':['knn3','knn4','strat']}
for nm,ks in combos.items():
    p=np.mean([L[k] for k in ks],axis=0)
    print(nm,'LOO mse %.4g rmse(k) %.1f'%(np.mean((p-ytr)**2),np.sqrt(np.mean((p-ytr)**2))*2601))
