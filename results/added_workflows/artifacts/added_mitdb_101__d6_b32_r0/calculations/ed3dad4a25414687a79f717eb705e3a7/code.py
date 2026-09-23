
import itertools, numpy as np
masks = [''.join(b) for b in itertools.product('01', repeat=6)]
even = [m for m in masks if m.count('1') % 2 == 0]
odd  = [m for m in masks if m.count('1') % 2 == 1]
free = {'000000','111111'}
buy_even = [m for m in even if m not in free]
print("num even", len(even), "num odd", len(odd), "buy_even", len(buy_even))
extra = ['111000','000111']
purchase = buy_even + extra
print("total purchases", len(purchase))
print(purchase)

# check orthogonality of the 32 even-point design for 22-param model
E = np.array([[int(c) for c in m] for m in even], dtype=float)
n = len(even)
cols = [np.ones(n)]
for j in range(6): cols.append(E[:,j])
for i in range(6):
    for j in range(i+1,6):
        cols.append(E[:,i]*E[:,j])
X = np.column_stack(cols)
XtX = X.T@X
print("XtX diag", np.diag(XtX))
print("offdiag max abs", np.max(np.abs(XtX - np.diag(np.diag(XtX)))))
