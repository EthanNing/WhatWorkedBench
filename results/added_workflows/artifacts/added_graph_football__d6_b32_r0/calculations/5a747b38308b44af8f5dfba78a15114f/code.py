
import json
design=[]
for x in range(64):
    b=format(x,'06b')
    a,c,d,e,f = [int(ch) for ch in b[:5]]
    if (a^c^d^e^f)==int(b[5]):
        design.append(b)
print(len(design))
extra=[m for m in design if m not in ('000000','111111')]
print(len(extra))
print(json.dumps(extra))
