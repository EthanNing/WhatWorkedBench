"""Gaussian-process reconstruction and variance-based acquisition."""
import numpy as np
def canonical(family,mask):
    if len(mask)!=6 or set(mask)-{'0','1'}:raise ValueError('Expected six mask bits')
    bits=list(mask)
    if family=='regression':
        if bits[2]=='0':bits[4]='0'
        if bits[1]=='0':bits[5]='0'
    elif family=='clustering':
        if bits[2]=='0':bits[4]='0'
    elif family=='image_restoration':
        if bits[1]=='0':bits[4]='0'
        if bits[0]=='0':bits[5]='0'
    elif family=='retrieval':
        if bits[4]=='1':bits[2]='0'
    elif family not in ('classification','forecasting'):
        raise ValueError('Unknown family')
    return ''.join(bits)

LENGTHS=(.25,.5,1.,2.,4.,8.)

NUGGETS=(1e-6,.001,.03,.3)

def geometry(dimensions,family=None):
    masks=[format(i,f'0{dimensions}b') for i in range(2**dimensions)]
    representatives=[canonical(family,m) if family is not None else m for m in masks]
    x=np.array([[int(v) for v in mask] for mask in representatives])
    distances=np.sum(x[:,None,:]!=x[None,:,:],axis=2)
    edge_low=[];edge_high=[]
    for bit in range(dimensions):
        for i in range(len(masks)):
            if not i&(1<<bit):edge_low.append(i);edge_high.append(i|(1<<bit))
    return {'masks':masks,'representatives':representatives,
            'kernels':{length:np.exp(-distances/length) for length in LENGTHS},
            'edge_low':np.array(edge_low),'edge_high':np.array(edge_high)}

def posterior(geo,observations):
    masks=geo['masks'];ids=[masks.index(m) for m in observations];y=np.array(list(observations.values()))
    choices=[]
    for length,full in geo['kernels'].items():
        K=full[np.ix_(ids,ids)]
        for nugget in NUGGETS:
            inverse=np.linalg.inv(K+nugget*np.eye(len(ids)))
            z=inverse@np.ones(len(ids));normalizer=float(np.sum(z))
            precision=inverse-np.outer(z,z)/normalizer
            residuals=(precision@y)/np.diag(precision)
            loss=float(np.mean(residuals**2))
            choices.append((round(loss,14),-length,-nugget,length,nugget,inverse,z,normalizer,loss))
    selected=min(choices,key=lambda row:row[:3])
    _,_,_,length,nugget,inverse,z,normalizer,loss=selected
    full=geo['kernels'][length];cross=full[:,ids]
    intercept=float(z@y/normalizer)
    mean=intercept+cross@(inverse@(y-intercept))
    mean_correction=1-cross@z
    covariance=full-cross@inverse@cross.T+np.outer(mean_correction,mean_correction)/normalizer
    covariance=(covariance+covariance.T)/2
    predictions=dict(zip(masks,map(float,np.clip(mean,0,1))))
    predictions.update(observations)
    return predictions,covariance,{'lengthscale':length,'nugget':nugget,'observed_loo_mse':loss}

def acquisition(geo,covariance,indices,objective,nugget):
    columns=covariance[:,indices]
    if objective=='grid_variance':numerator=np.sum(columns**2,axis=0)
    elif objective=='effect_variance':
        numerator=np.sum((columns[geo['edge_high']]-columns[geo['edge_low']])**2,axis=0)
    else:raise ValueError('Unknown acquisition objective')
    # Numerator/denominator is the exact one-point posterior trace reduction
    # under the currently fitted kernel. Nugget is regularization, not a claim
    # that deterministic native outcomes contain observational noise.
    return numerator/np.maximum(np.diag(covariance)[indices]+nugget,1e-12)

def episode(dimensions,budget,oracle,objective='effect_variance',family=None):
    geo=geometry(dimensions,family);masks=geo['masks'];observed={m:oracle(m) for m in (masks[0],masks[-1])}
    path=[]
    for _ in range(budget):
        known={geo['representatives'][masks.index(m)] for m in observed}
        candidates=[i for i,m in enumerate(masks) if m==geo['representatives'][i] and m not in known]
        if not candidates:break
        _,covariance,parameters=posterior(geo,observed)
        gain=acquisition(geo,covariance,candidates,objective,parameters['nugget'])
        position=min(range(len(candidates)),key=lambda k:(-round(float(gain[k]),12),candidates[k]))
        mask=masks[candidates[position]];observed[mask]=oracle(mask)
        path.append({'mask':mask,**parameters,'expected_trace_reduction':float(gain[position])})
    predictions,_,parameters=posterior(geo,observed)
    # Known code equivalences may cover unmeasured masks exactly. These are
    # predictions inferred from public code, not additional measured receipts.
    known={geo['representatives'][masks.index(m)]:u for m,u in observed.items()}
    for i,m in enumerate(masks):
        if geo['representatives'][i] in known:predictions[m]=known[geo['representatives'][i]]
    return predictions,observed,path,parameters
