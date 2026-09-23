
import numpy as np, itertools, json
from wwb_helpers import fit_surface
meas={o["mask"]:o["utility"] for o in observations}
Delta5=meas["111111"]-meas["111110"]
def Delta(k):
    return 0.0 if k<=1 else Delta5*(k-1)/4.0
struct={}
for bits in itertools.product([0,1],repeat=6):
    m=''.join(map(str,bits)); k=sum(bits[:5])
    if m in meas: struct[m]=meas[m]; continue
    p=m[:5]+str(sum(bits[:5])%2)
    struct[m]=meas[p]+(bits[5]-int(p[5]))*Delta(k)
unobs=[m for m in struct if m not in meas]
print("unobserved",len(unobs))
for deg in (1,2):
    r=fit_surface(observations,degree=deg)["predictions"]
    diffs=[abs(r[m]-struct[m]) for m in unobs]
    print(f"deg{deg}: MAE vs structural {np.mean(diffs):.4f} max {np.max(diffs):.4f}")
    for m in ["000010","000100","000111","110010","100011","011101","000110"]:
        print("   ",m,"helper %.4f struct %.4f"%(r[m],struct[m]))
