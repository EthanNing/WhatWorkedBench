
import os
print(os.listdir('/job'))
for f in os.listdir('/job'):
    p=os.path.join('/job',f)
    if os.path.isfile(p):
        print("==",f,os.path.getsize(p))
        try:
            t=open(p).read()
            print(t[:1500])
        except Exception as e: print("err",e)
