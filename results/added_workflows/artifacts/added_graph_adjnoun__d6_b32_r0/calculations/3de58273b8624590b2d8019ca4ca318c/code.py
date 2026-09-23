
import numpy as np, json
from wwb_helpers import fit_surface
for deg in (1,2):
    r=fit_surface(observations, degree=deg)
    print("deg",deg,type(r))
    if isinstance(r,dict):
        for m in ["000000","000001","000010","000111","110010","100011","011101"]:
            print("   ",m,round(r[m],4))
    else:
        print("   sample",np.round(np.asarray(r)[:5],4))
