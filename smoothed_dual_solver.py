import numpy as np
import time
def objective(x,c,q): return float(c@x+.5*np.dot(q,x*x))
def x_mu(lam,A,c,q,u,mu,x0):
    return np.clip((mu*x0-c-A.T@lam)/(q+mu),0,u)
def dual(lam,A,b,c,q,u):
    s=c+A.T@lam;x=np.zeros_like(c);pos=q>0
    x[pos]=np.clip(-s[pos]/q[pos],0,u[pos])
    x[~pos]=np.where(s[~pos]<0,u[~pos],0)
    return objective(x,s,q)-float(lam@b)
def solve(A,b,c,q,u,mu,K=5000,accelerated=False,log_every=20):
    if mu<=0 or K<1:raise ValueError("mu>0 and K>=1 required")
    n,m=A.shape;x0=np.zeros(m);lam=np.zeros(n);y=lam.copy();t=1.
    L=float(np.linalg.norm(A/np.sqrt(q+mu),2)**2);alpha=1/L
    avg=np.zeros(m);best=-np.inf;history=[];start=time.perf_counter()
    for k in range(K):
        z=y if accelerated else lam
        x=x_mu(z,A,c,q,u,mu,x0);nxt=z+alpha*(A@x-b);nxt-=nxt.mean()
        if accelerated:
            tn=(1+np.sqrt(1+4*t*t))/2
            y=nxt+((t-1)/tn)*(nxt-lam);t=tn
        lam=nxt;avg+=(x-avg)/(k+1)
        if k==0 or (k+1)%log_every==0 or k==K-1:
            lb=dual(lam,A,b,c,q,u);best=max(best,lb)
            last=x_mu(lam,A,c,q,u,mu,x0)
            history.append(dict(iteration=k+1,seconds=time.perf_counter()-start,
              dual_lb=lb,best_dual_lb=best,
              last_feasibility=float(np.linalg.norm(A@last-b)),
              avg_feasibility=float(np.linalg.norm(A@avg-b)),
              avg_objective_uncertified=objective(avg,c,q)))
    return dict(history=history,lam=lam,x_avg=avg,alpha=alpha,L=L,
                seconds=time.perf_counter()-start)
