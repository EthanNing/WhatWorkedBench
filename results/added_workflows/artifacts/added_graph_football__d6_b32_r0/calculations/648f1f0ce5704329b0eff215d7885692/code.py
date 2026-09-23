
import json, numpy as np
obs={o["mask"]:o["utility"] for o in observations}
masks=[format(i,'06b') for i in range(64)]
def nbrs(m):
    b=[int(c) for c in m]; out=[]
    for i in range(6):
        b2=b.copy(); b2[i]^=1; out.append(''.join(map(str,b2)))
    return out
single={'100000':0.8046797541146143,'010000':0.8247075153678366,'001000':0.8134047194130478,
        '000100':0.8135369158569634,'000010':0.6422103245422696,'000001':0.5}
final={}
for m in masks:
    if m in obs: final[m]=obs[m]
    elif m in single: final[m]=single[m]
    else: final[m]=float(np.median([obs[k] for k in nbrs(m)]))
s1=json.dumps(final,separators=(',',':'))
print(repr(s1))
print(json.dumps(s1))
