"""Public data-only numerical references, also copied into the agent scratchpad.

No benchmark files, complete outcomes, dataset loaders, model APIs or hidden
state are referenced here. The caller supplies every observation.
"""
import itertools
import math
import numpy as np

def _observed(observations):
    if not isinstance(observations,list) or not observations:raise ValueError('Supply nonempty list of mask/utility observations')
    result={}
    for row in observations:
        mask=row['mask'];value=row['utility']
        if not isinstance(mask,str) or not 2<=len(mask)<=8 or set(mask)-{'0','1'}:raise ValueError('Invalid mask')
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid utility')
        if mask in result and result[mask]!=value:raise ValueError('Conflicting measurements')
        result[mask]=float(value)
    m=len(next(iter(result)))
    if any(len(mask)!=m for mask in result):raise ValueError('Inconsistent mask sizes')
    return result,m

def features(m,degree=1):
    if degree not in (1,2):raise ValueError('Reference degree must be 1 or 2')
    masks=[f'{i:0{m}b}' for i in range(2**m)]
    bits=np.array([[2*int(v)-1 for v in mask] for mask in masks],float)
    X=np.column_stack([np.ones(len(masks))]+[np.prod(bits[:,indices],axis=1) for d in range(1,degree+1) for indices in itertools.combinations(range(m),d)])
    return masks,X

def _ridge(X,y,alpha):
    mean=X[:,1:].mean(axis=0);centered=X[:,1:]-mean
    beta=np.linalg.solve(centered.T@centered+alpha*np.eye(centered.shape[1]),centered.T@(y-y.mean()))
    return np.r_[y.mean()-mean@beta,beta]

def fit_surface(observations,degree=1):
    """Return a complete bounded prediction table using LOO-selected ridge.

    degree=1 uses main effects, degree=2 adds pair interactions. LOO diagnostics
    concern observed values only and are NOT hidden-truth accuracy estimates.
    """
    observed,m=_observed(observations);masks,X=features(m,degree)
    ids=[masks.index(mask) for mask in observed];y=np.array(list(observed.values()))
    if len(ids)<2:raise ValueError('At least two observations required')
    grid=[1e-5,.001,.1,1.,10.]
    losses={a:float(np.mean([(float(X[ids[i]]@_ridge(X[[idx for j,idx in enumerate(ids) if j!=i]],np.delete(y,i),a))-y[i])**2 for i in range(len(ids))])) for a in grid}
    alpha=min(grid,key=lambda a:(round(losses[a],14),-a))
    prediction=np.clip(X@_ridge(X[ids],y,alpha),0,1)
    return {'predictions':{**dict(zip(masks,map(float,prediction))),**observed},'degree':degree,
            'ridge_alpha':alpha,'observed_loo_mse':losses[alpha],'observed_loo_mse_by_alpha':losses,
            'observations_used':len(observed),'hidden_outcomes_used':0}

def suggest_experiments(observations,count=1,degree=1):
    """Greedy variance design, conditional on existing measured masks.

    Uses a fixed ridge precision, not an oracle or utility-dependent lookahead.
    You may choose another design or modify recommendations yourself.
    """
    observed,m=_observed(observations);masks,X=features(m,degree);chosen=[masks.index(mask) for mask in observed]
    if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=len(masks)-len(chosen):raise ValueError('Invalid number of new experiments')
    result=[]
    for _ in range(count):
        covariance=np.linalg.inv(X[chosen].T@X[chosen]+np.diag([1e-8]+[.01]*(X.shape[1]-1)))
        index=min((i for i in range(len(masks)) if i not in chosen),key=lambda i:(-round(float(X[i]@covariance@X[i]),10),i))
        chosen.append(index);result.append(masks[index])
    return result
