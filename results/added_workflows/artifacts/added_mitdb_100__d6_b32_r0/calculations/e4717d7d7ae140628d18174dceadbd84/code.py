
import numpy as np
obs = {
"000000":0.9986468200270636,"000011":0.9906040268456375,"000101":0.9932705248990579,
"000110":0.9945945945945946,"010001":1.0,"010010":0.9879518072289156,
"010100":0.9787798408488063,"010111":0.997289972899729}
masks=list(obs.keys())
e={m:738*(1-obs[m])/obs[m] for m in masks}
for m in masks: print(m,"e=",round(e[m],4))
def z(b): return 1 if b=='1' else -1
# columns B,D,E,F  (positions 1,3,4,5)
for pos,nm in [(1,'B'),(3,'D'),(4,'E'),(5,'F')]:
    vals=[z(m[pos]) for m in masks]
    ev=[e[m] for m in masks]
    print(nm,"direct coef = mean(z*e) =", round(np.mean(np.array(vals)*np.array(ev)),4),
          " (mean_plus,mean_minus)=",round(np.mean([v for v,s in zip(ev,vals) if s>0]),3),
          round(np.mean([v for v,s in zip(ev,vals) if s<0]),3))
X=np.array([[1,z(m[1]),z(m[3]),z(m[4]),z(m[5])] for m in masks],float)
y=np.array([e[m] for m in masks])
print("lstsq coef:",np.round(np.linalg.lstsq(X,y,rcond=None)[0],4))
print("mean e:",round(y.mean(),4))
