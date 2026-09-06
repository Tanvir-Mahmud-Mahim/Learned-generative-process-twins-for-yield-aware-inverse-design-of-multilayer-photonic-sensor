"""R2 revision (R1.4/R2.4): pass/fail yield counts, Wilson 95% CIs, and a
(tau_stop, tau_pass) threshold sweep for the nominal and FabGAN-robust
designs, using the same masks and paired draws as the original evaluation."""
import json, os, sys
import numpy as np, jax.numpy as jnp
sys.path.insert(0,'.')
from src import tmm_jax, process
from experiments import common
stop,pas=[np.asarray(m).astype(bool) for m in tmm_jax.band_masks(common.CENTER)]
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n)
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return [float((c-h)/d),float((c+h)/d)]
yld=json.load(open('results/exp5_yield.json'))
K=5000; out={}
for k in ["nominal","fabgan"]:
    d=np.array(yld[k]["d"]); n=np.array(yld[k]["n"])
    rng2=np.random.default_rng(common.SEED+123)          # paired across designs
    D,N=process.corrupt_ensemble(d,n,K,rng2)
    T=np.asarray(tmm_jax.transmittance_batch(jnp.asarray(D),jnp.asarray(N)))
    ml=T[:,stop].max(axis=1); mp=T[:,pas].mean(axis=1)
    grid={}
    for ts in [0.06,0.09,0.12,0.15]:
        for tp in [0.85,0.90,0.92]:
            kk=int(((ml<=ts)&(mp>=tp)).sum())
            grid["%.2f_%.2f"%(ts,tp)]={"pass":kk,"n":K,"yield":kk/K,"ci95":wilson(kk,K)}
    out[k]=grid
json.dump(out,open("results/r2_yield.json","w"),indent=1)
p=out["nominal"]["0.09_0.90"]; q=out["fabgan"]["0.09_0.90"]
print("paper spec (9%,90%): nominal %d/%d CI%s | fabgan %d/%d CI%s"%(p["pass"],p["n"],
  [round(x,4) for x in p["ci95"]],q["pass"],q["n"],[round(x,4) for x in q["ci95"]]))
print("DONE")
