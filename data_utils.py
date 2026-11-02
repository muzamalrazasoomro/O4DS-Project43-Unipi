import numpy as np
def make_instance(n=20,m=60,seed=42):
    if n<2 or m<n-1: raise ValueError("Invalid graph size")
    rng=np.random.default_rng(seed)
    arcs=[]
    for v in range(1,n):
        w=int(rng.integers(v));arcs.append((w,v) if rng.random()<.5 else (v,w))
    for _ in range(m-n+1):
        a,b=rng.choice(n,2,replace=False);arcs.append((int(a),int(b)))
    A=np.zeros((n,m))
    for j,(a,b) in enumerate(arcs): A[a,j]=-1;A[b,j]=1
    u=rng.uniform(1,5,m);xf=u*rng.uniform(.2,.8,m);b=A@xf
    c=rng.uniform(-1,1,m);q=rng.uniform(.1,2,m)
    q[rng.choice(m,max(1,m//5),replace=False)]=0
    return A,b,c,q,u,xf
