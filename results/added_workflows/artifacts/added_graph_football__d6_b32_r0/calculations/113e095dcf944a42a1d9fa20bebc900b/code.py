
import numpy as np, itertools, json
obs = {
"000000":0.5,"111111":0.8803952673673078,
"000011":0.6422103245422696,"000101":0.8135369158569634,"000110":0.8915658668781811,
"001001":0.8134047194130478,"001010":0.8839645713530306,"001100":0.8134047194130478,
"001111":0.8817833300284222,"010001":0.8247075153678366,"010010":0.8856500760129553,
"010100":0.821105162271135,"010111":0.8825104104699583,"011000":0.8164121885121289,
"011011":0.8809901513649283,"011101":0.8240134840372794,"011110":0.8835679820212836,
"100001":0.8046797541146143,"100010":0.8779826822658471,"100100":0.8134047194130478,
"100111":0.8805274638112235,"101000":0.8134047194130478,"101011":0.8788750082622777,
"101101":0.8134047194130478,"101110":0.8786767135964043,"110000":0.8241456804811951,
"110011":0.8787097627073832,"110101":0.8244100733690264,"110110":0.8782140260426994,
"111001":0.8242778769251107,"111010":0.8785445171524886,"111100":0.8165443849560446}

def bits(m): return np.array([int(c) for c in m])
def xcode(m): return 2*bits(m)-1

masks = [format(i,'06b') for i in range(64)]
X = np.array([xcode(m) for m in masks], float)  # 64x6
# design matrix: intercept, 6 mains, 15 pairs
def design(cols):
    n=cols.shape[0]
    D=[np.ones(n)]
    for i in range(6): D.append(cols[:,i])
    for i,j in itertools.combinations(range(6),2): D.append(cols[:,i]*cols[:,j])
    return np.column_stack(D)
D = design(X)
names=['int']+[f"m{i}" for i in range(6)]+[f"p{i}{j}" for i,j in itertools.combinations(range(6),2)]

obs_masks=list(obs.keys())
Di=np.array([design(xcode(m)[None,:])[0] for m in obs_masks])
yi=np.array([obs[m] for m in obs_masks])
coef,res,rank,sv=np.linalg.lstsq(Di,yi,rcond=None)
pred_i=Di@coef
print("residuals at 32 design pts: max abs",np.max(np.abs(pred_i-yi)),"rms",np.sqrt(np.mean((pred_i-yi)**2)))
for m,r in sorted(zip(obs_masks,pred_i-yi),key=lambda t:-abs(t[1]))[:10]:
    print(m, round(r,5))
print("coef:")
for n,c in zip(names,coef): print(n, round(c,5))
