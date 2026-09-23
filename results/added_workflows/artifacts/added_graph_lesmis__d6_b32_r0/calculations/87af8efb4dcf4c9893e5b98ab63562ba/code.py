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
sub=[o for o in obs_list if o['mask']!='000000']
mtr=[o['mask'] for o in sub]; ytr=np.array([o['utility'] for o in sub]); n=len(ytr)
B=np.array([bvec(m) for m in mtr])  # cols A B C D E F
def ridge_loo(X,y,alpha):
    Xc=X-X.mean(0); yc=y-y.mean()
    A=Xc.T@Xc+alpha*np.eye(X.shape[1]); H=Xc@np.linalg.solve(A,Xc.T)
    r=yc-H@yc; return np.mean((r/(1-np.diag(H)))**2)
def CV(make,reps=40,folds=6,seed=3):
    rng=np.random.default_rng(seed); errs=[]
    for _ in range(reps):
        idx=rng.permutation(n)
        for f in np.array_split(idx,folds):
            tr=np.setdiff1d(idx,f)
            Xtr,Xte=make(B[tr]),make(B[f])
            mx=Xtr.mean(0);my=ytr[tr].mean()
            w=np.linalg.solve((Xtr-mx).T@(Xtr-mx)+1.0*np.eye(Xtr.shape[1]),(Xtr-mx).T@(ytr[tr]-my))
            errs.append(((Xte-mx)@w+my-ytr[f])**2)
    return np.concatenate(errs).mean()
A,Bc,C,D,E,F=[B[:,i] for i in range(6)]
def mk_mains(X): return np.column_stack([X[:,i] for i in range(6)])
def mk_eint(X): 
    cols=[X[:,i] for i in range(6)]+[X[:,4]*X[:,j] for j in (0,1,2,3,5)]
    return np.column_stack(cols)
def mk_efint(X): 
    cols=[X[:,i] for i in range(6)]+[X[:,4]*X[:,j] for j in (0,1,2,3,5)]+[X[:,5]*X[:,j] for j in (0,1,2,3)]
    return np.column_stack(cols)
def mk_strat(X):
    e=X[:,4]; f=X[:,5]
    cols=[]
    for ev in (0,1):
        for fv in (0,1):
            ind=((e==ev)&(f==fv)).astype(float)
            cols+= [ind, ind*X[:,0], ind*X[:,1], ind*X[:,2], ind*X[:,3]]
    return np.column_stack(cols)
def fullp(X,deg):
    cols=[]
    for dd in range(1,deg+1):
        for c in itertools.combinations(range(6),dd): cols.append(np.prod(X[:,c],axis=1))
    return np.column_stack(cols)
print('mains     %.4g'%CV(mk_mains))
print('eint      %.4g'%CV(mk_eint))
print('efint     %.4g'%CV(mk_efint))
print('strat     %.4g'%CV(mk_strat))
print('deg2      %.4g'%CV(lambda X:fullp(X,2)))
print('deg3      %.4g'%CV(lambda X:fullp(X,3)))
