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
mtr=[o['mask'] for o in sub]; ytr=np.array([o['utility'] for o in sub])
allm=[''.join(b) for b in itertools.product('01',repeat=6)]
Btr=np.array([bvec(m) for m in mtr]); Ball=np.array([bvec(m) for m in allm])
def fullp(X,deg):
    cols=[]
    for dd in range(1,deg+1):
        for c in itertools.combinations(range(6),dd): cols.append(np.prod(X[:,c],axis=1))
    return np.column_stack(cols)
def fitpred(Xtr,ytr,Xte,alpha):
    mx=Xtr.mean(0);my=ytr.mean()
    w=np.linalg.solve((Xtr-mx).T@(Xtr-mx)+alpha*np.eye(Xtr.shape[1]),(Xtr-mx).T@(ytr-my))
    return (Xte-mx)@w+my
P={}
P['deg2']=fitpred(fullp(Btr,2),ytr,fullp(Ball,2),10)
P['deg1']=fitpred(fullp(Btr,1),ytr,fullp(Ball,1),10)
# knn distance
from sklearn.neighbors import KNeighborsRegressor
for k in (3,4,5):
    m=KNeighborsRegressor(k,weights='distance').fit(Btr,ytr); P['knn%d'%k]=m.predict(Ball)
# eint/efint
def mke(X): return np.column_stack([X[:,i] for i in range(6)]+[X[:,4]*X[:,j] for j in (0,1,2,3,5)]+[X[:,5]*X[:,j] for j in (0,1,2,3)])
P['efint']=fitpred(mke(Btr),ytr,mke(Ball),1.0)
# strat
def mks(X):
    e=X[:,4];f=X[:,5];cols=[]
    for ev in (0,1):
        for fv in (0,1):
            ind=((e==ev)&(f==fv)).astype(float)
            cols+=[ind,ind*X[:,0],ind*X[:,1],ind*X[:,2],ind*X[:,3]]
    return np.column_stack(cols)
P['strat']=fitpred(mks(Btr),ytr,mks(Ball),1.0)
P['ens']=np.mean([P['deg2'],P['knn3'],P['strat'],P['efint']],axis=0)
# neighbor distances for unseen
seen=set(o['mask'] for o in obs_list)
un=[m for m in allm if m not in seen]
Buno=np.array([bvec(m) for m in un])
D=(Buno[:,None,:]!=Btr[None,:,:]).sum(2)
mind=D.min(1)
print('unseen count',len(un))
print('nearest-observed hamming distance distribution:',np.bincount(mind))
print()
print('mask  kdeg2  kdeg1  kknn3  kknn4  kstrat kefint  kens   minD')
for i,m in enumerate(un):
    print('%s %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f   %d'%(m,P['deg2'][allm.index(m)]*2601,P['deg1'][allm.index(m)]*2601,P['knn3'][allm.index(m)]*2601,P['knn4'][allm.index(m)]*2601,P['strat'][allm.index(m)]*2601,P['efint'][allm.index(m)]*2601,P['ens'][allm.index(m)]*2601,mind[i]))
