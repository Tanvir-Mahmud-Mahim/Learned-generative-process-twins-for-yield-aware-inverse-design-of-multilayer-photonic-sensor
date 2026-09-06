"""Baselines (full Gaussian, 5-comp GMM) scored on the identical 12-design
protocol of exp_r2_seeds, for a like-for-like variability comparison."""
import json, os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0,'.')
from src import tmm_jax, process, fabgan
from experiments import common
from experiments.exp3_fabgan import load_inputs
STOP,PAS=tmm_jax.band_masks(common.CENTER)
def tails(J):
    Js=np.sort(J); q=max(1,int(np.ceil(0.05*len(J))))
    return float(np.percentile(J,5)), float(Js[:q].mean())
(rd,rn,fd,fn,X,C,stop,pas,cal_d,cal_n,cal_jnom,jr_low)=load_inputs()
bench=np.load(os.path.join(common.DATA,"benchmark_designs.npz"))
used=set(map(tuple,np.round(rd[::common.RUNS_PER_RECIPE],9)))
held=[i for i in range(bench["d_um"].shape[0]) if tuple(np.round(bench["d_um"][i],9)) not in used][:12]
ref={}
for i in held:
    rng=np.random.default_rng(common.SEED+900+i)
    D,Nn=process.corrupt_ensemble(bench["d_um"][i],bench["n0"][i],400,rng)
    ref[i]=tails(np.asarray(tmm_jax.notch_merit_batch(jnp.asarray(D),jnp.asarray(Nn),STOP,PAS)))
mu=X.mean(0); Cv=np.cov(X.T)+1e-9*np.eye(X.shape[1]); L=np.linalg.cholesky(Cv)
from sklearn.mixture import GaussianMixture
gm=GaussianMixture(5,covariance_type='full',reg_covar=1e-5,random_state=0).fit(X)
out={}
for name in ["gauss_full","gmm5"]:
    p5e,cve=[],[]
    for i in held:
        d,n=bench["d_um"][i],bench["n0"][i]
        rng=np.random.default_rng(common.SEED+900+i)
        if name=="gauss_full":
            xs=mu+rng.standard_normal((400,X.shape[1]))@L.T
        else:
            xs,_=gm.sample(400)
            xs=xs[rng.permutation(400)]
        Df=d*(1+xs[:,:20]); Nf=n+xs[:,20:]
        p5g,cvg=tails(np.asarray(tmm_jax.notch_merit_batch(jnp.asarray(Df),jnp.asarray(Nf),STOP,PAS)))
        p5e.append(abs(ref[i][0]-p5g)); cve.append(abs(ref[i][1]-cvg))
    out[name]={"P5_err":float(np.mean(p5e)),"CVaR5_err":float(np.mean(cve))}
    print(name,out[name])
json.dump(out,open("results/r2_seedsbase.json","w"),indent=1); print("DONE")
