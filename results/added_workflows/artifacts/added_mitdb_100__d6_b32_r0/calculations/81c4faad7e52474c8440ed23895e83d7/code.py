
masks_even = [format(i,'06b') for i in range(64) if format(i,'06b').count('1')%2==0]
print(len(masks_even))
print(masks_even)
to_buy = [m for m in masks_even if m not in ('000000','111111')]
print(len(to_buy))
print(to_buy)
