
import itertools, json
# Resolution-VI half fraction: F = A^B^C^D^E  (defining relation I = ABCDEF)
principal=[]
for bits in itertools.product([0,1],repeat=5):
    a,b,c,d,e=bits
    f=a^b^c^d^e
    principal.append(''.join(str(x) for x in (a,b,c,d,e,f)))
print("principal count:", len(principal))
print("contains 000000:", "000000" in principal, " contains 111111:", "111111" in principal)
print(json.dumps(principal))
# complementary fraction points (have odd total parity)
print("complement extras:", ['000001','111110'])
