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
names='ABCDEF'
def bvec(m): return np.array([int(c) for c in m],float)
# training set for g: nonempty-feature masks observed (exclude 000000); add 000001=0.5 known
train=[o for o in obs_list if o['mask']!='000000']
# note 000001 not observed but logically 0.5; keep it out of fit as check later; include optional
def make_design(masks, deg):
    B=np.array([bvec(m) for m in masks])
    cols=[np.ones(len(masks))]
    idx=list(range(6))
    for d in range(1,deg+1):
        for combo in itertools.combinations(range(6),d):
            cols.append(np.prod(B[:,combo],axis=1))
    return np.column_stack(cols)

def ridge_loo(X,y,alpha):
    n=len(y)
    Xc=X-X.mean(0); yc=y-y.mean()
    # do not penalize intercept; center handles it
    XtX=Xc.T@Xc+alpha*np.eye(X.shape[1])
    XtX[0,0]-=alpha  # unpenalized intercept column is zero after centering anyway
    H=Xc@np.linalg.solve(XtX,Xc.T)
    resid=yc-H@yc
    loo=resid/(1-np.diag(H))
    return np.mean(loo**2), Xc, yc

mtr=[o['mask'] for o in train]; ytr=np.array([o['utility'] for o in train])
print('train n=',len(ytr))
for deg in (1,2,3,4):
    X=make_design(mtr,deg)
    best=(1e9,None)
    for a in [0.0001,0.001,0.01,0.03,0.1,0.3,1,3,10,30,100]:
        m,_,_=ridge_loo(X,ytr,a)
        if m<best[0]: best=(m,a)
    print('deg',deg,'p=',X.shape[1],'best LOO mse=%.6g alpha=%s'%best)
