import numpy as np, itertools
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsRegressor
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
def bvec(m): return np.array([int(c) for c in m],float)
sub=[o for o in obs_list if o['mask']!='000000']
mtr=[o['mask'] for o in sub]; ytr=np.array([o['utility'] for o in sub]); n=len(ytr)
B=np.array([bvec(m) for m in mtr])
print('var y=%.4g  mean y=%.4g'%(ytr.var(),ytr.mean()))
def full(B,deg):
    cols=[]
    for dd in range(1,deg+1):
        for c in itertools.combinations(range(6),dd): cols.append(np.prod(B[:,c],axis=1))
    return np.column_stack(cols)
X1=full(B,1);X2=full(B,2);X3=full(B,3)
rng=np.random.default_rng(7)
def cv(mk,XX,reps=60,folds=6):
    errs=[]
    for r in range(reps):
        idx=rng.permutation(n)
        for f in np.array_split(idx,folds):
            tr=np.setdiff1d(idx,f); m=mk(); m.fit(XX[tr],ytr[tr])
            errs.append((m.predict(XX[f])-ytr[f])**2)
    return np.concatenate(errs).mean()
print('deg1 ridge %.4g'%cv(lambda:RidgeCV(alphas=np.logspace(-3,3,25)),X1))
print('deg2 ridge %.4g'%cv(lambda:RidgeCV(alphas=np.logspace(-3,3,25)),X2))
print('deg3 ridge %.4g'%cv(lambda:RidgeCV(alphas=np.logspace(-3,3,25)),X3))
for k in (2,3,4,5,6,8):
    print('knn%d dist %.4g'%(k,cv(lambda k=k:KNeighborsRegressor(k,weights='distance'),B,reps=1,folds=n)))
print('knn4 unif %.4g'%cv(lambda:KNeighborsRegressor(4),B,reps=1,folds=n))
