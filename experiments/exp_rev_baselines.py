"""Twin-fidelity comparison across FIVE process models on ONE identical
held-out-design protocol (reviewer 1 #3): diagonal Gaussian, full-covariance
Gaussian, Gaussian mixture (GMM, 5 comp.), nonparametric bootstrap, FabGAN.
Metrics: |dP5|, |dCVaR5|, W1(J) vs the withheld true process. Resumable.
"""
import json, os, sys, time, pickle
import numpy as np, jax, jax.numpy as jnp
from scipy.stats import wasserstein_distance
from sklearn.mixture import GaussianMixture
sys.path.insert(0,'.')
from src import tmm_jax, process, fabgan
from experiments import common

STATE="results/rev_baselines_state.pkl"; STOP,PAS=tmm_jax.band_masks(common.CENTER)
def ystat(J):
    Js=np.sort(J); q=max(1,int(np.ceil(0.05*len(J)))); return float(np.percentile(J,5)),float(Js[:q].mean())
def merit(D,N):
    D=np.clip(D,0.5*tmm_jax.T_LO,2*tmm_jax.T_HI); N=np.clip(N,1.4,2.6)
    return np.asarray(tmm_jax.notch_merit_batch(jnp.asarray(D),jnp.asarray(N),STOP,PAS))

def main():
    t0=time.time()
    dat=np.load(os.path.join(common.DATA,"process_traces.npz"))
    rd,rn,fd,fn=dat["recipe_d"],dat["recipe_n"],dat["fab_d"],dat["fab_n"]
    X=fabgan.errors_from_traces(rd,rn,fd,fn)
    bench=np.load(os.path.join(common.DATA,"benchmark_designs.npz"))
    used=set(map(tuple,np.round(rd[::common.RUNS_PER_RECIPE],9)))
    held=[i for i in range(bench["d_um"].shape[0]) if tuple(np.round(bench["d_um"][i],9)) not in used][:30]
    Hd,Hn=bench["d_um"][held],bench["n0"][held]
    gt=pickle.load(open('results/gauss_twins.pkl','rb'))
    gd,gf=gt["gauss_diag"],gt["gauss_full"]
    gp=jax.tree.map(jnp.asarray,pickle.load(open('results/fabgan_vanilla.pkl','rb')))
    gmm=GaussianMixture(n_components=5,covariance_type="full",reg_covar=1e-5,random_state=0).fit(X)

    names=["gauss_diag","gauss_full","gmm","boot","fabgan"]
    st=pickle.load(open(STATE,'rb')) if os.path.exists(STATE) else {"i":0,"acc":{m:{"p5":[],"cv":[],"w":[]} for m in names}}
    acc=st["acc"]; K=400
    for i in range(st["i"],len(held)):
        d,n=Hd[i],Hn[i]
        rng=np.random.default_rng(common.SEED+900+i)         # per-design common seed
        Dt,Nt=process.corrupt_ensemble(d,n,K,rng); Jt=merit(Dt,Nt); p5t,cvt=ystat(Jt)
        samp={}
        xg=gd.sample(rng,d,n,K); samp["gauss_diag"]=xg
        samp["gauss_full"]=gf.sample(rng,d,n,K)
        xm=gmm.sample(K)[0]; samp["gmm"]=(d*(1+xm[:,:20]),n+xm[:,20:])
        xb=X[rng.integers(0,X.shape[0],K)]; samp["boot"]=(d*(1+xb[:,:20]),n+xb[:,20:])
        k=jax.random.PRNGKey(common.SEED+900+i); samp["fabgan"]=fabgan.sample_fabgan(gp,k,d,n,K)
        for m in names:
            Dm,Nm=samp[m]; Jm=merit(Dm,Nm); p5m,cvm=ystat(Jm)
            acc[m]["p5"].append(abs(p5t-p5m)); acc[m]["cv"].append(abs(cvt-cvm)); acc[m]["w"].append(wasserstein_distance(Jt,Jm))
        st["i"]=i+1
        if time.time()-t0>32: pickle.dump(st,open(STATE,'wb')); print("PROGRESS",i+1,"/",len(held)); return
    out={m:{"P5_abs_err_mean":float(np.mean(acc[m]["p5"])),
            "CVaR5_abs_err_mean":float(np.mean(acc[m]["cv"])),
            "induced_perf_W1_mean":float(np.mean(acc[m]["w"]))} for m in names}
    json.dump(out,open('results/rev_baselines.json','w'),indent=1)
    if os.path.exists(STATE): os.remove(STATE)
    print(json.dumps(out,indent=1)); print("DONE")

if __name__=="__main__": main()
