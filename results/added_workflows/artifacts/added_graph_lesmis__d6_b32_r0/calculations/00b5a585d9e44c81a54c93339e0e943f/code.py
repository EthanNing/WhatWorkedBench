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
B=np.array([bvec(m) for m in mtr])
def full(B,deg):
    cols=[]
    for dd in range(1,deg+1):
        for c in itertools.combinations(range(6),dd): cols.append(np.prod(B[:,c],axis=1))
    return np.column_stack(cols)
def ridge_loo(X,y,alpha):
    Xc=X-X.mean(0); yc=y-y.mean()
    A=Xc.T@Xc+alpha*np.eye(X.shape[1]); H=Xc@np.linalg.solve(A,Xc.T)
    r=yc-H@yc; return r/(1-np.diag(H))
X1,X2,X3=full(B,1),full(B,2),full(B,3)
print('mask   y_k   r1   r2a1  r2a10  r3a10  (k units)')
res={}
for nm,X,a in [('d1',X1,1),('d2a1',X2,1),('d2a10',X2,10),('d3a10',X3,10),('d3a1',X3,1)]:
    res[nm]=ridge_loo(X,ytr,a)*2601
masks=[''.join(b) for b in itertools.product('01',repeat=6)]
idx=[masks.index(m) for m in mtr]
for i,m in enumerate(mtr):
    print('%s %5.0f  %6.1f %6.1f %6.1f %6.1f %6.1f'%(m,ytr[i]*2601,res['d1'][i],res['d2a1'][i],res['d2a10'][i],res['d3a10'][i],res['d3a1'][i]))
for nm in res:
    print(nm,'LOO rmse (k units) %.2f'%np.sqrt(np.mean(res[nm]**2)))
