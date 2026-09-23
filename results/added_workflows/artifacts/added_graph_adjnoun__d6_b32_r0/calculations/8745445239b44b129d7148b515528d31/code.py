
import numpy as np
from wwb_helpers import fit_surface
r=fit_surface(observations, degree=2)
print("keys sample:",list(r.keys())[:5])
print("n keys",len(r))
print(open('/job/wwb_helpers.py').read()[:3000])
