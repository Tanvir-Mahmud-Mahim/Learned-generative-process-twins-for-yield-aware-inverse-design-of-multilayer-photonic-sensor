"""R2 revision: training variability and learning curves (R1.2/R2.2).
(a) 5 GAN seeds on the standard 400 traces; (b) 3 independently generated
trace datasets; (c) learning curve M in {50,100,200,400} x 3 seeds.
Metric: mean |dP5|, |dCVaR5| over 12 held-out designs x 400 true draws.
Resumable: rerun until DONE."""
import json, os, sys, time, pickle
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0,'.')
from src import tmm_jax, process, fabgan
from experiments import common
from experiments.exp3_fabgan import load_inputs
STATE="results/r2_seeds_state.pkl"; STOP,PAS=tmm_jax.band_masks(common.CENTER)
def tails(J):
    Js=np.sort(J); q=max(1,int(np.ceil(0.05*len(J))))
    return float(np.percentile(J,5)), float(Js[:q].mean())
def fidelity(gp, held, bench, ref):
    p5e,cve=[],[]
    for i in held:
        d,n=bench["d_um"][i],bench["n0"][i]
        p5t,cvt=ref[i]
        k=jax.random.PRNGKey(common.SEED+900+i)
        Df,Nf=fabgan.sample_fabgan(gp,k,d,n,400)
        p5g,cvg=tails(np.asarray(tmm_jax.notch_merit_batch(jnp.asarray(Df),jnp.asarray(Nf),STOP,PAS)))
        p5e.append(abs(p5t-p5g)); cve.append(abs(cvt-cvg))
    return float(np.mean(p5e)), float(np.mean(cve))
def main():
    t0=time.time(); budget=lambda: time.time()-t0>34
    (rd,rn,fd,fn,X,C,stop,pas,cal_d,cal_n,cal_jnom,jr_low)=load_inputs()
    bench=np.load(os.path.join(common.DATA,"benchmark_designs.npz"))
    used=set(map(tuple,np.round(rd[::common.RUNS_PER_RECIPE],9)))
    held=[i for i in range(bench["d_um"].shape[0]) if tuple(np.round(bench["d_um"][i],9)) not in used][:12]
    st=pickle.load(open(STATE,'rb')) if os.path.exists(STATE) else {"ref":{},"res":{"seed":{},"data":{},"curve":{}}}
    # reference true tails per held design (computed once)
    for i in held:
        if i in st["ref"]: continue
        rng=np.random.default_rng(common.SEED+900+i)
        D,Nn=process.corrupt_ensemble(bench["d_um"][i],bench["n0"][i],400,rng)
        st["ref"][i]=tails(np.asarray(tmm_jax.notch_merit_batch(jnp.asarray(D),jnp.asarray(Nn),STOP,PAS)))
        pickle.dump(st,open(STATE,'wb'))
        if budget(): print("PROG ref",i); return
    Rd=rd[::common.RUNS_PER_RECIPE]; Rn=rn[::common.RUNS_PER_RECIPE]
    jobs=[]
    for s in [7,17,27,37,47]: jobs.append(("seed",str(s),{"X":None,"C":None,"seed":s,"M":400}))
    for ds in [1,2,3]: jobs.append(("data",str(ds),{"dsseed":ds,"seed":7,"M":400}))
    for M in [50,100,200,400]:
        for s in [7,17,27]: jobs.append(("curve",f"{M}_{s}",{"seed":s,"M":M}))
    for kind,key,cfg in jobs:
        if key in st["res"][kind]: continue
        w=st.setdefault("work",{})
        if st.get("work_key")!=(kind,key):
            st["work"]={}; st["work_key"]=(kind,key); w=st["work"]
        if "X" not in w:
            if kind=="data":
                rng=np.random.default_rng(4000+cfg["dsseed"])
                rd2,rn2,fd2,fn2=process.trace_dataset(Rd,Rn,common.RUNS_PER_RECIPE,rng)
                w["X"]=fabgan.errors_from_traces(rd2,rn2,fd2,fn2)
                w["C"]=np.asarray(fabgan.norm_recipe(jnp.asarray(rd2),jnp.asarray(rn2)))
            else:
                M=cfg["M"]; w["X"]=X[:M]; w["C"]=C[:M]
            pickle.dump(st,open(STATE,'wb'))
            if budget(): print("PROG traces",kind,key); return
        gp,dp,out=fabgan.train_fabgan(jax.random.PRNGKey(cfg["seed"]),w["C"],w["X"],cal_d,cal_n,
            cal_jnom,jr_low,stop,pas,steps=1500,tail_w=0.0,
            resume=w.get("gr"),max_seconds=max(5,32-(time.time()-t0)))
        if gp is None:
            w["gr"]=out; pickle.dump(st,open(STATE,'wb')); print("PROG gan",kind,key,out["t"]); return
        p5,cv=fidelity(jax.tree.map(jnp.asarray,gp),held,bench,st["ref"])
        st["res"][kind][key]={"P5_err":p5,"CVaR5_err":cv,**{k:v for k,v in cfg.items() if k!="X"}}
        st["work"]={}; st["work_key"]=None; pickle.dump(st,open(STATE,'wb'))
        print("done",kind,key,round(p5,4),round(cv,4))
        if budget(): return
    json.dump(st["res"],open("results/r2_seeds.json","w"),indent=1)
    os.remove(STATE); print("DONE")
if __name__=="__main__": main()
