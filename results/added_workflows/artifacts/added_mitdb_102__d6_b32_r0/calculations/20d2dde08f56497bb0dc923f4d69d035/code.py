
from wwb_helpers import fit_surface, suggest_experiments
obs = [{"mask": "000000", "utility": 0.9541284403669725}, {"mask": "111111", "utility": 1.0}]
s = suggest_experiments(obs, count=32, degree=2)
print(type(s))
print(s)
