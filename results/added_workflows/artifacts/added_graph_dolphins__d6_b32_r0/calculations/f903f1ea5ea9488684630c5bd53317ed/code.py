
from wwb_helpers import suggest_experiments, fit_surface
print("observations:", observations)
try:
    sug = suggest_experiments(observations, count=32, degree=2)
    print("suggest 32:", sug)
except Exception as e:
    print("err count=32:", repr(e))
    sug = suggest_experiments(observations, count=1, degree=2)
    print("suggest 1:", sug)
