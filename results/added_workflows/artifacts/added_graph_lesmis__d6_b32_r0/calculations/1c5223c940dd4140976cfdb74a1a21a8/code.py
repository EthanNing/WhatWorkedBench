from wwb_helpers import suggest_experiments, fit_surface
obs=[{"mask":"000000","utility":0.5},{"mask":"111111","utility":0.9019607843137254}]
s=suggest_experiments(obs, count=32, degree=2)
print(type(s))
print(s)
