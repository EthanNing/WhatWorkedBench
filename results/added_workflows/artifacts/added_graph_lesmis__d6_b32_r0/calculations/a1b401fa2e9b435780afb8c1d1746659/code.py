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
mtr=[o['mask'] for o in sub]; ytr=np.array([o['utility'] for o in sub]); ktr=np.round(ytr*2601)
B=np.array([bvec(m) for m in mtr])
nm='ABCDEF'
# OLS quadratic
T=[(), ] 
def tname(c): return ''.join(nm[j] for j in c) if c else '1'
cols=[np.ones(33)]
Tn=['1']
for dd in range(1,3):
    for c in itertools.combinations(range(6),dd):
        cols.append(np.prod(B[:,c],axis=1)); Tn.append(tname(c))
X=np.column_stack(cols)
beta,res,rank,sv=np.linalg.lstsq(X,ytr,rcond=None)
pred=X@beta; r=ytr-pred
print('quadratic in-sample max|r|=%.3g rmse=%.3g (in k units max %.1f)'%(np.abs(r).max(),r.std(),np.abs(r).max()*2601))
order=np.argsort(-np.abs(r))
for i in order[:12]:
    print('  ',mtr[i],'y=%.4f pred=%.4f r=%.4f (%.1f k)'%(ytr[i],pred[i],r[i],r[i]*2601))
print('coefs (k units):')
for n,b in zip(Tn,beta):
    print('   %-4s %+7.2f'%(n,b*2601))
