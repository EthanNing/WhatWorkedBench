"""Native CPU operations with explicit legal component bindings."""
from __future__ import annotations
import math
import numpy as np
from scipy.ndimage import gaussian_filter,median_filter
from scipy.signal import wiener
from sklearn.cluster import KMeans
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import balanced_accuracy_score,mean_absolute_error,normalized_mutual_info_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer,PolynomialFeatures,StandardScaler
import unicodedata

def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


FACTORS={
    'classification':{'A':'sign(x)*log1p(abs(x)) feature transform','B':'train-fitted standard scaling',
                      'C':'balanced instead of uniform class weights','D':'LogisticRegression C=0.05 instead of C=1'},
    'regression':{'A':'log1p training target and expm1 prediction inverse','B':'standard scaling before feature construction',
                  'C':'degree-two polynomial features instead of original features','D':'Ridge alpha=10 instead of alpha=1'},
    'clustering':{'A':'sign(x)*log1p(abs(x)) feature transform','B':'feature-only standard scaling',
                  'C':'PCA retaining 95 percent variance instead of all original features','D':'10 KMeans initializations instead of 1'},
    'forecasting':{'A':'linear time trend feature','B':'lags 1..12 instead of lag 1 only',
                   'C':'log1p series transform and expm1 prediction inverse','D':'Ridge alpha=10 instead of alpha=1'},
    'image_restoration':{'A':'3x3 median filtering','B':'Gaussian filtering with sigma=0.8',
                         'C':'local Wiener filtering with a 5x5 neighborhood and automatically estimated noise power',
                         'D':'unsharp masking, Gaussian sigma=1 and amount=0.5'},
    'retrieval':{'A':'unigrams plus bigrams instead of unigrams','B':'sklearn English stop words removed',
                 'C':'sublinear term frequency instead of raw counts','D':'IDF weighting instead of none'},
}

def switches(mask):
    if not isinstance(mask,str) or len(mask)!=4 or set(mask)-{'0','1'}:raise ValueError('Expected four-bit configuration')
    return tuple(v=='1' for v in mask)

def signed_log(x):return np.sign(x)*np.log1p(np.abs(x))

def supervised_pipeline(family,mask):
    a,b,c,d=switches(mask);steps=[]
    if family=='classification' and a:steps.append(('signed_log',FunctionTransformer(signed_log)))
    if b:steps.append(('scale',StandardScaler()))
    if family=='classification':
        steps.append(('model',LogisticRegression(C=.05 if d else 1.,class_weight='balanced' if c else None,
                                                  solver='lbfgs',max_iter=3000,random_state=17)))
        return Pipeline(steps)
    if c:steps.append(('poly',PolynomialFeatures(2,include_bias=False)))
    steps.append(('model',Ridge(alpha=10. if d else 1.,solver='svd')))
    model=Pipeline(steps)
    return TransformedTargetRegressor(regressor=model,func=np.log1p,inverse_func=np.expm1) if a else model

def cluster(x,n_clusters,mask):
    a,b,c,d=switches(mask);steps=[]
    if a:steps.append(('signed_log',FunctionTransformer(signed_log)))
    if b:steps.append(('scale',StandardScaler()))
    if c:steps.append(('pca',PCA(n_components=.95,svd_solver='full')))
    steps.append(('model',KMeans(n_clusters=n_clusters,n_init=10 if d else 1,random_state=17,max_iter=300,algorithm='lloyd')))
    return Pipeline(steps).fit_predict(x)

def forecast(series,cut,mask):
    a,b,c,d=switches(mask)
    values=np.log1p(series) if c else np.array(series,copy=True)
    times=np.arange(12,len(values))
    lag_count=12 if b else 1
    features=np.column_stack([values[times-lag] for lag in range(1,lag_count+1)])
    if a:features=np.column_stack([features,times/float(cut)])
    train=times<cut
    model=Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=10. if d else 1.,solver='svd'))])
    model.fit(features[train],values[times[train]])
    prediction=model.predict(features[~train])
    return np.expm1(prediction) if c else prediction

def restore(noisy,mask):
    a,b,c,d=switches(mask);image=np.array(noisy,copy=True)
    if a:image=median_filter(image,size=3,mode='reflect')
    if b:image=gaussian_filter(image,sigma=.8,mode='reflect')
    if c:image=wiener(image,mysize=(5,5))
    if d:image=image+.5*(image-gaussian_filter(image,sigma=1.,mode='reflect'))
    return np.clip(image,0,1)

def retrieve(documents,queries,exclude_self,mask):
    a,b,c,d=switches(mask)
    docs=sorted(documents,key=lambda r:r['doc_id'])
    vectorizer=TfidfVectorizer(ngram_range=(1,2) if a else (1,1),stop_words='english' if b else None,
                               sublinear_tf=c,use_idf=d,norm='l2',dtype=np.float64)
    matrix=vectorizer.fit_transform([r['text'] for r in docs])
    query_matrix=vectorizer.transform([text for _,text in queries])
    similarities=(query_matrix@matrix.T).toarray()
    normalized=[normalized_text(r['text']) for r in docs] if exclude_self else None
    result={}
    for (qid,text),row in zip(queries,similarities):
        if not np.isfinite(row).all():raise ValueError('Nonfinite retrieval scores')
        order=np.argsort(-row,kind='stable')
        if exclude_self:
            query=normalized_text(text);order=[i for i in order if normalized[i]!=query]
        result[qid]=[docs[i]['doc_id'] for i in order[:10]]
    return result

def ndcg(rankings,qrels):
    values=[]
    for qid,ranked in rankings.items():
        gold=qrels[qid]
        if len(ranked)!=10 or len(set(ranked))!=10:raise ValueError('Invalid top ten rankings')
        dcg=math.fsum(gold.get(doc,0)/math.log2(rank+2) for rank,doc in enumerate(ranked))
        ideal=math.fsum(gain/math.log2(rank+2) for rank,gain in enumerate(sorted(gold.values(),reverse=True)[:10]))
        values.append(dcg/ideal)
    return float(np.mean(values))

def evaluate(family,data,mask):
    extra={}
    if family in ('classification','regression'):
        model=supervised_pipeline(family,mask);model.fit(data['x_train'],data['y_train']);pred=model.predict(data['x_eval'])
        if family=='classification':
            native=float(balanced_accuracy_score(data['y_eval'],pred));utility=native;metric='balanced_accuracy'
        else:
            native=float(mean_absolute_error(data['y_eval'],pred));scale=float(np.quantile(data['y_train'],.75)-np.quantile(data['y_train'],.25))
            utility=1/(1+native/scale);metric='MAE';extra['normalization_scale']=scale
    elif family=='clustering':
        pred=cluster(data['x'],data['n_clusters'],mask)
        native=float(normalized_mutual_info_score(data['y'],pred,average_method='arithmetic'));utility=native;metric='NMI'
    elif family=='forecasting':
        pred=forecast(data['series'],data['cut'],mask)
        native=float(mean_absolute_error(data['series'][data['cut']:],pred))
        scale=float(np.quantile(data['series'][:data['cut']],.75)-np.quantile(data['series'][:data['cut']],.25))
        utility=1/(1+native/scale);metric='one_step_MAE';extra['normalization_scale']=scale
    elif family=='image_restoration':
        pred=restore(data['noisy'],mask)
        native=float(np.sqrt(np.mean((data['clean']-pred)**2)));utility=1/(1+native/.1);metric='pixel_RMSE'
        extra['normalization_scale']=.1
    elif family=='retrieval':
        pred=retrieve(data['documents'],data['queries'],data['exclude_self'],mask)
        native=ndcg(pred,data['qrels']);utility=native;metric='NDCG@10'
    else:raise ValueError('Unknown workflow')
    if not isinstance(pred,dict) and not np.isfinite(pred).all():raise ValueError('Nonfinite native predictions')
    if not math.isfinite(utility) or not 0<=utility<=1:raise ValueError('Invalid bounded utility')
    return pred,{'utility':utility,'native_metric':metric,'native_value':native,'valid_prediction':True,**extra}
