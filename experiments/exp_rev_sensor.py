"""Sensor-level metrics and pass/fail yield (reviewer 1 #2, #17; reviewer 2).
Computes, for the nominal and FabGAN-robust designs at 532 nm:
  - T(532 nm), max stop-band leakage, mean pass-band transmission, edge steepness
  - pass/fail yield P(max_stop <= tau_stop AND mean_pass >= tau_pass) under the
    withheld true process, with common draws.
"""
import json, os, sys
import numpy as np, jax.numpy as jnp
sys.path.insert(0,'.')
from src import tmm_jax, process
from experiments import common

CENTER=common.CENTER
lam=np.asarray(tmm_jax.LAM_UM)*1000.0
stop,pas=tmm_jax.band_masks(CENTER)
stop=np.asarray(stop).astype(bool); pas=np.asarray(pas).astype(bool)

def metrics(d,n):
    T=np.asarray(tmm_jax.transmittance_jit(jnp.asarray(d),jnp.asarray(n)))
    t532=float(np.interp(532.0,lam,T))
    maxleak=float(T[stop].max())
    meanpass=float(T[pas].mean())
    return t532,maxleak,meanpass,T

yld=json.load(open('results/exp5_yield.json'))
out={}
for k in ["nominal","fabgan"]:
    d=np.array(yld[k]["d"]); n=np.array(yld[k]["n"])
    t532,maxleak,meanpass,T=metrics(d,n)
    out[k]={"T_532_nm":t532,"max_stopband_leakage":maxleak,"mean_passband_T":meanpass}

# nominal-anchored spec thresholds (rounded from the nominal design's clean values)
tau_stop=round(out["nominal"]["max_stopband_leakage"]+0.03,2)  # allowance above clean
tau_pass=0.90
out["_spec"]={"tau_stop":tau_stop,"tau_pass":tau_pass,
              "definition":"pass iff max stop-band T <= tau_stop and mean pass-band T >= tau_pass"}

# pass/fail yield under true process, common draws
K=5000; rng=np.random.default_rng(common.SEED+77)
draws=[process.corrupt(np.array(yld["nominal"]["d"]),np.array(yld["nominal"]["n"]),rng) for _ in range(0)]  # warm
for k in ["nominal","fabgan"]:
    d=np.array(yld[k]["d"]); n=np.array(yld[k]["n"])
    rng2=np.random.default_rng(common.SEED+123)  # common seed across designs
    D,N=process.corrupt_ensemble(d,n,K,rng2)
    Tall=np.asarray(tmm_jax.transmittance_batch(jnp.asarray(D),jnp.asarray(N)))
    maxleak=Tall[:,stop].max(axis=1)
    meanp=Tall[:,pas].mean(axis=1)
    passed=(maxleak<=tau_stop)&(meanp>=tau_pass)
    out[k]["yield_passfail"]=float(passed.mean())
    out[k]["yield_n"]=int(K)
json.dump(out,open('results/rev_sensor.json','w'),indent=1)
print(json.dumps(out,indent=1))
