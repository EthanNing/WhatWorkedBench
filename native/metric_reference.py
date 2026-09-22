import math
import numpy as np
from sklearn.metrics import ndcg_score

def independent_metric(family,data,prediction,row):
    if family=='classification':
        y=data['y_eval'];classes=sorted(set(y))
        return math.fsum(sum(int(p==g) for p,g in zip(prediction,y) if g==c)/sum(y==c) for c in classes)/len(classes)
    if family in ('regression','forecasting'):
        y=data['y_eval'] if family=='regression' else data['series'][data['cut']:]
        return math.fsum(abs(float(p)-float(g)) for p,g in zip(prediction,y))/len(y)
    if family=='image_restoration':
        return math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(prediction.flat,data['clean'].flat))/prediction.size)
    if family=='clustering':
        y=data['y'];pred=prediction;n=len(y)
        classes=sorted(set(y));clusters=sorted(set(pred))
        a={c:int(sum(y==c)) for c in classes};b={c:int(sum(pred==c)) for c in clusters}
        mi=0.
        for c in classes:
            for k in clusters:
                count=sum((y==c)&(pred==k))
                if count:mi+=count/n*math.log(count*n/(a[c]*b[k]))
        hy=-math.fsum(v/n*math.log(v/n) for v in a.values())
        hp=-math.fsum(v/n*math.log(v/n) for v in b.values())
        return 2*mi/(hy+hp) if hy+hp else 1.
    docs=sorted(data['documents'],key=lambda r:r['doc_id']);index={d['doc_id']:i for i,d in enumerate(docs)}
    labels=np.zeros((len(data['queries']),len(docs)));scores=np.zeros_like(labels)
    for i,(qid,_) in enumerate(data['queries']):
        for doc,gain in data['qrels'][qid].items():labels[i,index[doc]]=gain
        for rank,doc in enumerate(prediction[qid]):scores[i,index[doc]]=10-rank
    return float(ndcg_score(labels,scores,k=10))
