
import numpy as np
obs=[("000000",0.9541284403669725),("111111",1.0),("000001",0.9771812080536912),("000010",0.9667994687915007),("000011",0.9837837837837838),("000100",0.9567567567567568),("000110",0.7906976744186046),("001000",0.9688385269121813),("001010",0.5536121673003802),("001101",0.8426073131955485),("010000",0.9732620320855615),("010111",0.7805695142378559),("011001",1.0),("011011",0.6953199617956065),("011100",0.9972451790633609),("011101",0.9972451790633609),("011110",0.8845686512758202),("100000",0.9667994687915007),("100101",1.0),("100111",0.8481012658227848),("101000",0.9688385269121813),("101001",0.9688385269121813),("101010",0.5536121673003802),("101011",0.6854990583804144),("101100",0.8426073131955485),("101101",0.8426073131955485),("101110",0.9904761904761905),("110000",0.9837837837837838),("110001",1.0),("110011",1.0),("110100",1.0),("110110",0.8389154704944178),("111000",1.0),("111010",0.6195744680851064)]
masks=[format(i,'06b') for i in range(64)]
midx={m:i for i,m in enumerate(masks)}
X=np.array([[int(c) for c in m] for m in masks],float); Z1=2*X-1
def design(deg):
    cols=[np.ones(64)]
    from itertools import combinations
    for d in range(1,deg+1):
        for comb in combinations(range(6),d):
            col=np.ones(64)
            for j in comb: col=col*Z1[:,j]
            cols.append(col)
    return np.column_stack(cols)
obs_m=[midx[m] for m,u in obs]; y=np.array([u for m,u in obs])
for deg in [1,2,3]:
    Z=design(deg); Zd=Z[obs_m]
    # LOO
    n=len(y); loo=np.zeros(n)
    for i in range(n):
        tr=[j for j in range(n) if j!=i]
        b,_,_,_=np.linalg.lstsq(Zd[tr],y[tr],rcond=None)
        loo[i]=Zd[i]@b
    b,_,_,_=np.linalg.lstsq(Zd,y,rcond=None)
    pred=Zd@b
    print("deg",deg,"params",Zd.shape[1],"train rmse",round(np.sqrt(np.mean((pred-y)**2)),5),"loo rmse",round(np.sqrt(np.mean((loo-y)**2)),5))
