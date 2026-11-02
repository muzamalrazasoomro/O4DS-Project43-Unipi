import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_utils import make_instance
from smoothed_dual_solver import solve as solve_a1,objective
from general_solver import solve as solve_a2
def write_csv(path,rows):
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--nodes",type=int,default=20)
    ap.add_argument("--arcs",type=int,default=60)
    ap.add_argument("--iterations",type=int,default=5000)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    out=Path("results");out.mkdir(exist_ok=True)
    A,b,c,q,u,xf=make_instance(args.nodes,args.arcs,args.seed)
    ref=solve_a2(A,b,c,q,u);ub=objective(xf,c,q)
    pilot=solve_a1(A,b,c,q,u,.1,K=min(1000,args.iterations))
    R_est=max(float(np.linalg.norm(pilot["lam"])),1e-8)
    D=.5*float(u@u);mu0=np.linalg.norm(A,2)*R_est/np.sqrt(2*args.iterations*D)
    print("A2 objective:",ref["objective"],"A2 time:",ref["seconds"])
    print("Known feasible upper bound:",ub)
    print("Heuristic mu:",mu0,"(pilot R estimate is not certified)")
    summary=[]
    for mu in (mu0/3,mu0,3*mu0):
        for accel in (False,True):
            name="accelerated" if accel else "gradient"
            result=solve_a1(A,b,c,q,u,mu,K=args.iterations,accelerated=accel)
            hist=result["history"]
            for r in hist:
                r["certified_gap"]=ub-r["best_dual_lb"]
                r["reference_minus_lb"]=ref["objective"]-r["best_dual_lb"]
            tag=f"{name}_mu_{mu:.3e}"
            write_csv(out/f"history_{tag}.csv",hist)
            h=hist[-1]
            summary.append(dict(method=name,mu=mu,iterations=args.iterations,
               time_seconds=result["seconds"],best_dual_lb=h["best_dual_lb"],
               certified_upper_bound=ub,certified_gap=h["certified_gap"],
               a2_reference=ref["objective"],avg_feasibility=h["avg_feasibility"],
               last_feasibility=h["last_feasibility"]))
            it=[r["iteration"] for r in hist]
            fig,ax=plt.subplots(1,2,figsize=(11,4))
            ax[0].plot(it,[r["best_dual_lb"] for r in hist],label="Dual LB")
            ax[0].axhline(ref["objective"],ls="--",color="black",label="A2 reference")
            ax[0].axhline(ub,ls=":",color="gray",label="Feasible UB")
            ax[0].legend();ax[0].set_title("Bounds");ax[0].set_xlabel("Iteration")
            ax[1].semilogy(it,[max(r["last_feasibility"],1e-16) for r in hist],label="Last")
            ax[1].semilogy(it,[max(r["avg_feasibility"],1e-16) for r in hist],label="Average")
            ax[1].legend();ax[1].set_title("Feasibility");ax[1].set_xlabel("Iteration")
            for a in ax:a.grid(alpha=.3)
            fig.tight_layout();fig.savefig(out/f"{tag}.png",dpi=180);plt.close(fig)
            print(tag,"LB:",h["best_dual_lb"],"avg residual:",h["avg_feasibility"])
    write_csv(out/"summary.csv",summary)
    print("Saved results to",out.resolve())
if __name__=="__main__":main()
