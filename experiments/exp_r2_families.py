"""R2 revision: process-family robustness sweep (R1.1/R1.4/R2.3/R2.5).
For each hidden-process VARIANT: regenerate 400 traces, retrain the plain
WGAN twin and refit the full Gaussian, robustify the nominal 532-nm design
under each twin, and evaluate all designs on the VARIANT true process.
Resumable: rerun until DONE.
"""
import json, os, sys, time, pickle
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0,'.')
from src import tmm_jax, process, fabgan, adjoint
from experiments import common
from experiments.exp3_fabgan import load_inputs

STATE="results/r2_families_state.pkl"
VARIANTS={
 "base":    {},
 "no_rare":  {"P_FLAKE":0.0},
 "rare2x":   {"P_FLAKE":0.10},
 "severe2x": {"FLAKE_MU":0.16},
 "corr_low": {"RHO":0.3},
 "corr_high":{"RHO":0.85},
 "skew2x":   {"SIG_N":0.04},
}
DEFAULTS={k:getattr(process,k) for k in ["P_FLAKE","FLAKE_MU","RHO","SIG_N"]}
STOP,PAS=tmm_jax.band_masks(0.532)

def set_variant(p):
    for k,v in DEFAULTS.items(): setattr(process,k,v)
    for k,v in p.items(): setattr(process,k,v)

def stats(J):
    Js=np.sort(J); q=max(1,int(np.ceil(0.05*len(J))))
    return {"mean":float(J.mean()),"P5":float(np.percentile(J,5)),"CVaR5":float(Js[:q].mean())}

def main():
    t0=time.time(); budget=lambda: time.time()-t0>34
    inv=json.load(open('results/exp4_inverse.json'))
    d0=np.asarray(inv["theta_d"]); n0=np.asarray(inv["theta_n"])
    # base recipes for traces (dedup from released traces)
    dat=np.load(os.path.join(common.DATA,"process_traces.npz"))
    Rd=dat["recipe_d"][::common.RUNS_PER_RECIPE]; Rn=dat["recipe_n"][::common.RUNS_PER_RECIPE]
    (_,_,_,_,_,_,stop,pas,cal_d,cal_n,cal_jnom,jr_low)=load_inputs()
    st=pickle.load(open(STATE,'rb')) if os.path.exists(STATE) else {"res":{},"work":{}}
    for name,params in VARIANTS.items():
        if name in st["res"]: continue
        set_variant(params)
        w=st["work"].setdefault(name,{"stage":"traces"})
        if w["stage"]=="traces":
            rng=np.random.default_rng(1234)
            rd,rn,fd,fn=process.trace_dataset(Rd,Rn,common.RUNS_PER_RECIPE,rng)
            w["X"]=fabgan.errors_from_traces(rd,rn,fd,fn)
            w["C"]=np.asarray(fabgan.norm_recipe(jnp.asarray(rd),jnp.asarray(rn)))
            w["stage"]="gan"; pickle.dump(st,open(STATE,'wb'))
            if budget(): print("PROG",name,"traces"); return
        if w["stage"]=="gan":
            gp,dp,out=fabgan.train_fabgan(jax.random.PRNGKey(7),w["C"],w["X"],cal_d,cal_n,
                cal_jnom,jr_low,stop,pas,steps=3000,tail_w=0.0,
                resume=w.get("gan_resume"),max_seconds=max(5,32-(time.time()-t0)))
            if gp is None:
                w["gan_resume"]=out; pickle.dump(st,open(STATE,'wb')); print("PROG",name,"gan",out["t"]); return
            w["gp"]=jax.tree.map(np.asarray,gp); w.pop("gan_resume",None)
            mu=w["X"].mean(0); Cv=np.cov(w["X"].T)+1e-9*np.eye(w["X"].shape[1])
            w["mu"]=mu; w["L"]=np.linalg.cholesky(Cv)
            w["stage"]="rob_gan"; pickle.dump(st,open(STATE,'wb'))
            if budget(): print("PROG",name,"gan done"); return
        if w["stage"]=="rob_gan":
            gp=jax.tree.map(jnp.asarray,w["gp"])
            vg=adjoint.make_gan_objective(gp,STOP,PAS,256,0.05)
            dd,nn,out=adjoint.robustify(d0,n0,vg,fabgan.DIM_Z,256,steps=250,seed=0,
                resume=w.get("rg_resume"),max_seconds=max(5,32-(time.time()-t0)))
            if dd is None:
                w["rg_resume"]=out; pickle.dump(st,open(STATE,'wb')); print("PROG",name,"robgan"); return
            w["d_gan"],w["n_gan"]=dd,nn; w.pop("rg_resume",None)
            w["stage"]="rob_gauss"; pickle.dump(st,open(STATE,'wb'))
            if budget(): print("PROG",name,"robgan done"); return
        if w["stage"]=="rob_gauss":
            vg=adjoint.make_gauss_objective(w["mu"],w["L"],STOP,PAS,256,0.05)
            dd,nn,out=adjoint.robustify(d0,n0,vg,w["mu"].shape[0],256,steps=250,seed=0,
                resume=w.get("rq_resume"),max_seconds=max(5,32-(time.time()-t0)))
            if dd is None:
                w["rq_resume"]=out; pickle.dump(st,open(STATE,'wb')); print("PROG",name,"robgauss"); return
            w["d_gs"],w["n_gs"]=dd,nn; w.pop("rq_resume",None)
            w["stage"]="eval"; pickle.dump(st,open(STATE,'wb'))
            if budget(): print("PROG",name,"robgauss done"); return
        if w["stage"]=="eval":
            ev=w.setdefault("ev",{})
            for key,(dd,nn) in {"nominal":(d0,n0),"gauss_full":(w["d_gs"],w["n_gs"]),
                                 "fabgan":(w["d_gan"],w["n_gan"])}.items():
                if key in ev: continue
                rng=np.random.default_rng(common.SEED+5) # paired draws across designs
                D,Nn=process.corrupt_ensemble(dd,nn,2000,rng)
                J=np.asarray(tmm_jax.notch_merit_batch(jnp.asarray(D),jnp.asarray(Nn),STOP,PAS))
                ev[key]=stats(J); pickle.dump(st,open(STATE,'wb'))
                if budget(): print("PROG",name,"eval",key); return
            st["res"][name]={"params":params,**{k:ev[k] for k in ev}}
            st["work"].pop(name); pickle.dump(st,open(STATE,'wb'))
            print("variant done:",name, {k:round(ev[k]["CVaR5"],4) for k in ev})
            if budget(): return
    set_variant({})
    json.dump(st["res"],open("results/r2_families.json","w"),indent=1)
    os.remove(STATE); print("DONE")

if __name__=="__main__": main()
