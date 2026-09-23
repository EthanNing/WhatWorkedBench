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
def bvec(m): return np.array([int(c) for c in m],float)
mtr=[o['mask'] for o in obs_list]; ytr=np.array([o['utility'] for o in obs_list])
def terms(deg):
    out=[]
    for d in range(1,deg+1):
        for c in itertools.combinations(range(6),d): out.append(c)
    return out
def design(masks,deg):
    B=np.array([bvec(m) for m in masks]); T=terms(deg)
    return np.column_stack([np.prod(B[:,c],axis=1) for c in T])
def ridge_fit(X,y,alpha):
    mx=X.mean(0); my=y.mean()
    Xc=X-mx; yc=y-my
    w=np.linalg.solve(Xc.T@Xc+alpha*np.eye(X.shape[1]), Xc.T@yc)
    return w,mx,my
def ridge_pred(X,model):
    w,mx,my=model; return (X-mx)@w+my
rng=np.random.default_rng(0)
n=len(ytr)
for deg in (1,2,3,4,5):
    X=design(mtr,deg)
    for a in [1e-6,1e-4,1e-3,0.01,0.1,1,10,100]:
        errs=[]
        for rep in range(200):
            idx=rng.permutation(n); folds=np.array_split(idx,6)
            for f in folds:
                te=f; tr=np.setdiff1d(idx,f)
                m=ridge_fit(X[tr],ytr[tr],a)
                errs.append((ridge_pred(X[te],m)-ytr[te])**2)
        errs=np.concatenate(errs)
        print('deg',deg,'a',a,'6fold mse %.3g'%errs.mean())
    print()
