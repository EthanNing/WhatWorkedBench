import numpy as np, itertools, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, Matern
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
mtr=[o['mask'] for o in sub]; ytr=np.array([o['utility'] for o in sub])
B=np.array([bvec(m) for m in mtr])
def full_design(B,deg=6):
    n,d=B.shape; cols=[]
    for dd in range(1,deg+1):
        for c in itertools.combinations(range(d),dd): cols.append(np.prod(B[:,c],axis=1))
    return np.column_stack(cols)
X=full_design(B,6); print('dictionary',X.shape)
from sklearn.model_selection import KFold, cross_val_predict
kf=KFold(6,shuffle=True,random_state=0)
for name,mdl in [('lasso',LassoCV(cv=6,random_state=0,max_iter=100000)),('enet',ElasticNetCV(cv=6,random_state=0,max_iter=100000))]:
    p=cross_val_predict(mdl,X,ytr,cv=kf)
    print(name,'6fold mse %.4g'%np.mean((p-ytr)**2))
    mdl.fit(X,ytr)
    nz=np.where(np.abs(mdl.coef_)>1e-8)[0]
    print('  selected',[(tuple(np.array(list(itertools.combinations(range(6),dd)))[i]) if False else i) for i in []])
    # decode terms
    T=[]; 
    for dd in range(1,7):
        for c in itertools.combinations(range(6),dd): T.append(c)
    print('  nz terms:',[ (''.join('ABCDEF'[j] for j in T[i]),round(mdl.coef_[i],4)) for i in nz])
# GP on bits
for ls in [[1.0]*6,[2.0]*6,None]:
    k=C(0.01)*Matern([1.0]*6,nu=2.5)+WhiteKernel(1e-6)
    g=GaussianProcessRegressor(kernel=k,normalize_y=True,n_restarts_optimizer=5,random_state=0)
    p=cross_val_predict(g,B,ytr,cv=KFold(33,shuffle=False,random_state=0))
    print('GP matern ls',ls,'LOO mse %.4g'%np.mean((p-ytr)**2))
