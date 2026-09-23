
from wwb_helpers import fit_surface
obs=[{"mask":"000000","utility":0.9986468200270636},{"mask":"111111","utility":1.0},
{"mask":"000011","utility":0.9906040268456375},{"mask":"000101","utility":0.9932705248990579},
{"mask":"000110","utility":0.9945945945945946},{"mask":"001001","utility":1.0},
{"mask":"001010","utility":0.558667676003028},{"mask":"001100","utility":1.0},
{"mask":"001111","utility":1.0},{"mask":"010001","utility":1.0},
{"mask":"010010","utility":0.9879518072289156},{"mask":"010100","utility":0.9787798408488063},
{"mask":"010111","utility":0.997289972899729},{"mask":"011000","utility":1.0},
{"mask":"011011","utility":0.7242394504416094},{"mask":"011101","utility":1.0},
{"mask":"011110","utility":0.9986468200270636},{"mask":"100001","utility":1.0},
{"mask":"100010","utility":0.9986468200270636},{"mask":"100100","utility":1.0},
{"mask":"100111","utility":1.0},{"mask":"101000","utility":1.0},
{"mask":"101011","utility":0.7178988326848249},{"mask":"101101","utility":1.0},
{"mask":"101110","utility":1.0},{"mask":"110000","utility":1.0},{"mask":"110011","utility":1.0},
{"mask":"110101","utility":1.0},{"mask":"110110","utility":1.0},{"mask":"111001","utility":1.0},
{"mask":"111010","utility":0.5023825731790333},{"mask":"111100","utility":1.0},
{"mask":"011010","utility":0.5023825731790333},{"mask":"101010","utility":0.558667676003028}]
r=fit_surface(obs,degree=2)
print(type(r))
try:
    print(len(r)); print(list(r)[:10] if not isinstance(r,tuple) else "tuple")
except Exception as e: print(e)
print(repr(r)[:500])
