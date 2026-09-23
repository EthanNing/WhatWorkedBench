
import numpy as np, json
from wwb_helpers import fit_surface
obs=[(o["mask"],o["utility"]) for o in observations]
print("n obs",len(obs))
for deg in (1,2):
    r=fit_surface(obs, degree=deg)
    print("deg",deg,"type",type(r))
    try:
        keys=list(r.keys())[:3]; print(" sample keys",keys, list(r.items())[:3])
        d=r
    except Exception as e:
        print(" not dict",e); d=None
    if d is not None:
        # compare with our structural table in a few unobserved spots
        pass
