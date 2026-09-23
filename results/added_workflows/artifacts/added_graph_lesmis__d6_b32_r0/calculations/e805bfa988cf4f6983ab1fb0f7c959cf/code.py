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
mtr=[o['mask'] for o in sub]; ytr=np.array([o['utility'] for o in sub])
Btr=np.array([bvec(m) for m in mtr]); Ball=np.array([bvec(m) for m in allm])
def knn_stable(X,y,Xt,k):
    out=[]
    for j in range(len(Xt)):
        D=((X-Xt[j])**2).sum(1)
        order=np.lexsort((np.arange(len(X)),D))[:k]
        dd=D[order]; w=1.0/(dd+1e-9); w/=w.sum(); out.append((w*y[order]).sum())
    return np.array(out)
P=0.5*knn_stable(Btr,ytr,Ball,3)+0.5*knn_stable(Btr,ytr,Ball,4)
pred={}
for i,m in enumerate(allm):
    pred[m]=round(float(min(max(P[i],0.5),2346/2601)),6)
pred['000000']=0.5; pred['000001']=0.5
for m,u in obs.items(): pred[m]=round(float(u),6)
print(json.dumps(pred))
print(len(pred))
