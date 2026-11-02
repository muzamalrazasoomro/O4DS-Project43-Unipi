import cvxpy as cp
import numpy as np
import time
def solve(A,b,c,q,u):
    x=cp.Variable(len(c))
    problem=cp.Problem(cp.Minimize(.5*cp.sum(cp.multiply(q,cp.square(x)))+c@x),
                       [A@x==b,x>=0,x<=u])
    t=time.perf_counter()
    problem.solve(solver=cp.OSQP,eps_abs=1e-8,eps_rel=1e-8,max_iter=100000)
    if problem.status not in ("optimal","optimal_inaccurate"):
        raise RuntimeError(problem.status)
    return dict(objective=float(problem.value),x=np.asarray(x.value).ravel(),
                seconds=time.perf_counter()-t,status=problem.status)
