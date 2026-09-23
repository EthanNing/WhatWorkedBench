import numpy as np, itertools, json
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
allm=[''.join(b) for b in itertools.product('01',repeat=6)]
obs={o['mask']:o['utility'] for o in obs_list}
sub=[o for o in obs_list if o['mask']!='000000']
mtr=[o['mask'] for o in sub]; ytr=np.array([o['utility'] for o in sub]); n=len(ytr)
Btr=np.array([bvec(m) for m in mtr]); Ball=np.array([bvec(m) for m in allm])
def fullp(X,deg):
    cols=[]
    for dd in range(1,deg+1):
        for c in itertools.combinations(range(6),dd): cols.append(np.prod(X[:,c],axis=1))
    return np.column_stack(cols)
def knn_pred(Xtr,ytr,Xte,k=4):
    D=((Xtr[:,None,:]-Xte[None,:,:])**2).sum(2)  # squared euclid = hamming for binary
    out=np.zeros(len(Xte))
    for j in range(len(Xte)):
        order=np.argsort(D[:,j])[:k]; dd=D[order,j]
        w=1.0/(dd+1e-9); w=w/w.sum(); out[j]=(w*ytr[order]).sum()
    return out
def knn_loo(X,y,k=4):
    out=np.zeros(len(y))
    D=((X[:,None,:]-X[None,:,:])**2).sum(2)
    for i in range(len(y)):
        order=np.argsort(D[i])[1:k+1]; dd=D[i,order]
        w=1.0/(dd+1e-9); w=w/w.sum(); out[i]=(w*y[order]).sum()
    return out
def ridge_fitpred(Xtr,y,Xte,alpha):
    mx=Xtr.mean(0);my=y.mean()
    w=np.linalg.solve((Xtr-mx).T@(Xtr-mx)+alpha*np.eye(Xtr.shape[1]),(Xtr-mx).T@(y-my))
    return (Xte-mx)@w+my
def ridge_loo(X,y,alpha):
    out=np.zeros(len(y))
    for i in range(len(y)):
        tr=np.ones(len(y),bool);tr[i]=False
        out[i]=ridge_fitpred(X[tr],y[tr],X[i:i+1],alpha)[0]
    return out
X2tr=fullp(Btr,2); X2all=fullp(Ball,2)
L_kn4=knn_loo(Btr,ytr,4); L_d2=ridge_loo(X2tr,ytr,10)
def mk_s(X):
    e=X[:,4];f=X[:,5];cols=[]
    for ev in (0,1):
        for fv in (0,1):
            ind=((e==ev)&(f==fv)).astype(float); cols+=[ind,ind*X[:,0],ind*X[:,1],ind*X[:,2],ind*X[:,3]]
    return np.column_stack(cols)
L_st=ridge_loo(mk_s(Btr),ytr,1.0)
for nm,p in [('knn4',L_kn4),('deg2',L_d2),('strat',L_st),
             ('0.5knn4+0.5deg2',0.5*L_kn4+0.5*L_d2),
             ('0.5knn4+0.5strat',0.5*L_kn4+0.5*L_st),
             ('knn4+0.3deg2+0.3strat',(L_kn4+0.3*L_d2+0.3*L_st)/1.6)]:
    print(nm,'LOO rmse_k %.1f'%(np.sqrt(np.mean((p-ytr)**2))*2601))
P_kn4=knn_pred(Btr,ytr,Ball,4); P_d2=ridge_fitpred(X2tr,ytr,X2all,10); P_st=ridge_fitpred(mk_s(Btr),ytr,mk_s(Ball),1.0)
for nm,lam in [('kn4 only',1.0),('70/15/15',0.7)]:
    pass
final_kn = 0.7*P_kn4+0.15*P_d2+0.15*P_st
final_d2 = P_d2
# choose final: use knn4-heavy blend
pred={}
for i,m in enumerate(allm):
    v=final_kn[i]
    pred[m]=float(min(max(v,0.5),2346/2601))
pred['000000']=0.5; pred['000001']=0.5
for m,u in obs.items(): pred[m]=float(u)
print('sample')
for m in ['100000','001000','000100','001010','100110','111110','101111','110111']:
    print(m,round(pred[m],5),'k=%.0f'%(pred[m]*2601))
json.dumps(pred)
print(len(pred))
