"""Latent-dimension ablation (reviewer 1 #6): retrain the plain WGAN twin at
d_z in {16,32,48,64} and measure yield-tail fidelity vs the hidden process."""
import json, os, sys, time, pickle
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0,'.')
from src import tmm_jax, process, fabgan
from experiments import common
from experiments.exp3_fabgan import load_inputs
STATE="results/rev_latent_state.pkl"; STOP,PAS=tmm_jax.band_masks(common.CENTER)
def ystat(J):
    Js=np.sort(J); q=max(1,int(np.ceil(0.05*len(J)))); return float(np.percentile(J,5)),float(Js[:q].mean())
def main():
    t0=time.time()
    (rd,rn,fd,fn,X,C,stop,pas,cal_d,cal_n,cal_jnom,jr_low)=load_inputs()
    bench=np.load(os.path.join(common.DATA,"benchmark_designs.npz"))
    used=set(map(tuple,np.round(rd[::common.RUNS_PER_RECIPE],9)))
    held=[i for i in range(bench["d_um"].shape[0]) if tuple(np.round(bench["d_um"][i],9)) not in used][:12]
    dims=[16,32,48,64]
    st=pickle.load(open(STATE,'rb')) if os.path.exists(STATE) else {"res":{}}
    for dz in dims:
        if str(dz) in st["res"]: continue
        fabgan.DIM_Z=dz
        gp,dp,out=fabgan.train_fabgan(jax.random.PRNGKey(7),C,X,cal_d,cal_n,cal_jnom,jr_low,
                                      stop,pas,steps=1500,tail_w=0.0)
        p5e,cve=[],[]
        for i in held:
            d,n=bench["d_um"][i],bench["n0"][i]; rng=np.random.default_rng(common.SEED+900+i)
            Dt,Nt=process.corrupt_ensemble(d,n,400,rng); p5t,cvt=ystat(np.asarray(tmm_jax.notch_merit_batch(jnp.asarray(Dt),jnp.asarray(Nt),STOP,PAS)))
            k=jax.random.PRNGKey(common.SEED+900+i); Df,Nf=fabgan.sample_fabgan(gp,k,d,n,400)
            p5g,cvg=ystat(np.asarray(tmm_jax.notch_merit_batch(jnp.asarray(Df),jnp.asarray(Nf),STOP,PAS)))
            p5e.append(abs(p5t-p5g)); cve.append(abs(cvt-cvg))
        st["res"][str(dz)]={"P5_abs_err_mean":float(np.mean(p5e)),"CVaR5_abs_err_mean":float(np.mean(cve))}
        pickle.dump(st,open(STATE,'wb')); print("done dz",dz,st["res"][str(dz)])
        if time.time()-t0>33: print("PROGRESS"); return
    json.dump(st["res"],open("results/rev_latent.json","w"),indent=1)
    if os.path.exists(STATE): os.remove(STATE)
    print(json.dumps(st["res"],indent=1)); print("DONE")
if __name__=="__main__": main()
