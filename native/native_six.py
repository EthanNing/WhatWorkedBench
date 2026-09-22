import types
import numpy as np
import native_workflows as base
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import RobustScaler

def switches(mask):
    if not isinstance(mask,str) or len(mask)!=6 or set(mask)-{'0','1'}:
        raise ValueError('Expected six-bit configuration')
    return tuple(v=='1' for v in mask)


def supervised_pipeline(family,mask):
    a,b,c,d,e,f=switches(mask); steps=[]
    if family=='classification' and a:
        steps.append(('signed_log',base.FunctionTransformer(base.signed_log)))
    if b:
        steps.append(('scale',RobustScaler() if family=='regression' and f else base.StandardScaler()))
    if family=='classification':
        if e:steps.append(('selection',SelectPercentile(f_classif,percentile=50)))
        if f:steps.append(('pca',base.PCA(n_components=.95,svd_solver='full')))
        steps.append(('model',base.LogisticRegression(C=.05 if d else 1.,class_weight='balanced' if c else None,
                                                      solver='lbfgs',max_iter=3000,random_state=17)))
        return base.Pipeline(steps)
    if c:steps.append(('poly',base.PolynomialFeatures(2,include_bias=False,interaction_only=e)))
    steps.append(('model',base.Ridge(alpha=10. if d else 1.,solver='svd')))
    model=base.Pipeline(steps)
    return base.TransformedTargetRegressor(regressor=model,func=np.log1p,inverse_func=np.expm1) if a else model


def cluster(x,n_clusters,mask):
    a,b,c,d,e,f=switches(mask); steps=[]
    if a:steps.append(('signed_log',base.FunctionTransformer(base.signed_log)))
    if b:steps.append(('scale',base.StandardScaler()))
    if c:steps.append(('pca',base.PCA(n_components=.95,svd_solver='full',whiten=e)))
    steps.append(('model',base.KMeans(n_clusters=n_clusters,n_init=10 if d else 1,random_state=17,
                                     max_iter=300,algorithm='lloyd',init='random' if f else 'k-means++')))
    return base.Pipeline(steps).fit_predict(x)


def forecast(series,cut,mask):
    a,b,c,d,e,f=switches(mask)
    levels=np.log1p(series) if c else np.array(series,copy=True)
    values=np.r_[0.,np.diff(levels)] if f else levels
    times=np.arange(12,len(values)); lag_count=12 if b else 1
    features=np.column_stack([values[times-lag] for lag in range(1,lag_count+1)])
    if a:features=np.column_stack([features,times/float(cut)])
    train=times<cut
    model=base.Pipeline([('scale',RobustScaler() if e else base.StandardScaler()),
                         ('model',base.Ridge(alpha=10. if d else 1.,solver='svd'))])
    model.fit(features[train],values[times[train]])
    prediction=model.predict(features[~train])
    if f:prediction=prediction+levels[times[~train]-1]
    return np.expm1(prediction) if c else prediction


def restore(noisy,mask):
    a,b,c,d,e,f=switches(mask); image=np.array(noisy,copy=True)
    if a:image=base.median_filter(image,size=5 if f else 3,mode='reflect')
    if b:image=base.gaussian_filter(image,sigma=1.6 if e else .8,mode='reflect')
    if c:image=base.wiener(image,mysize=(5,5))
    if d:image=image+.5*(image-base.gaussian_filter(image,sigma=1.,mode='reflect'))
    return np.clip(image,0,1)


def retrieve(documents,queries,exclude_self,mask):
    a,b,c,d,e,f=switches(mask)
    docs=sorted(documents,key=lambda r:r['doc_id'])
    vectorizer=base.TfidfVectorizer(ngram_range=(1,2) if a else (1,1),stop_words='english' if b else None,
                                   sublinear_tf=c,use_idf=d,norm=None if f else 'l2',binary=e,dtype=np.float64)
    matrix=vectorizer.fit_transform([r['text'] for r in docs])
    similarities=(vectorizer.transform([text for _,text in queries])@matrix.T).toarray()
    normalized=[base.normalized_text(r['text']) for r in docs] if exclude_self else None
    result={}
    for (qid,text),row in zip(queries,similarities):
        if not np.isfinite(row).all():raise ValueError('Nonfinite retrieval scores')
        order=np.argsort(-row,kind='stable')
        if exclude_self:
            query=base.normalized_text(text); order=[i for i in order if normalized[i]!=query]
        result[qid]=[docs[i]['doc_id'] for i in order[:10]]
    return result

_scope = {**base.evaluate.__globals__, **{name: globals()[name] for name in ('supervised_pipeline', 'cluster', 'forecast', 'restore', 'retrieve')}}
evaluate = types.FunctionType(base.evaluate.__code__, _scope, 'evaluate_six_factors')
