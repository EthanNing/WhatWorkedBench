
import numpy as np, itertools
from wwb_helpers import fit_surface
meas={o["mask"]:o["utility"] for o in observations}
Delta5=meas["111111"]-meas["111110"]
def Delta(k): return 0.0 if k<=1 else Delta5*(k-1)/4.0
struct={}
for b in itertools.product([0,1],repeat=6):
    m=''.join(map(str,b));k=sum(b[:5])
    if m in meas: struct[m]=meas[m]
    else:
        p=m[:5]+str(sum(b[:5])%2); struct[m]=meas[p]+(b[5]-int(p[5]))*Delta(k)
edges=[(''.join(map(str,b)),''.join(map(str,bb))) for b in itertools.product([0,1],repeat=6) for bb in [b[:i]+(b[i]^1,)+b[i+1:] for i in range(6)]]
for deg in (1,2):
    r=fit_surface(observations,degree=deg)["predictions"]
    de=[abs((r[a]-r[b])-(struct[a]-struct[b])) for a,b in edges]
    print(f"deg{deg} edge-MAE vs structural {np.mean(de):.4f}  max {np.max(de):.4f}  obs-cells-exact: {all(r[m]==meas[m] for m in meas)}")
