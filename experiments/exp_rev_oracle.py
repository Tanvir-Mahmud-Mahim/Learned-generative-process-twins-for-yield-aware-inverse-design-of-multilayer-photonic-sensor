"""Oracle true-process robust design + mean-only bias-correction baseline
(reviewer 1 #8). Resumable: run until DONE.

- mean-only: pre-distort recipe by the fitted systematic bias from the traces
  (isolates bias correction from tail learning).
- oracle: derivative-free (Gaussian hill-climb, multi-restart) maximization of
  the TRUE-process CVaR5 with full simulator access; upper bound on achievable
  true-process CVaR5.  Regret_method = CVaR5(oracle) - CVaR5(method).
"""
import json, os, sys, time
import numpy as np, jax.numpy as jnp
sys.path.insert(0,'.')
from src import tmm_jax, process, fabgan
from experiments import common

CENTER=common.CENTER
stop,pas=tmm_jax.band_masks(CENTER)
TLO,THI,NLO,NHI=tmm_jax.T_LO,tmm_jax.T_HI,1.6,2.4
STATE="results/rev_oracle_state.pkl"

def cvar_true(d,n,K,seed,alpha=0.05):
    rng=np.random.default_rng(seed)
    D,N=process.corrupt_ensemble(np.asarray(d),np.asarray(n),K,rng)
    J=np.asarray(tmm_jax.notch_merit_batch(jnp.asarray(D),jnp.asarray(N),stop,pas))
    q=max(1,int(np.ceil(alpha*K)))
    return float(np.sort(J)[:q].mean())

def clip(d,n):
    return np.clip(d,TLO,THI), np.clip(n,NLO,NHI)

def main():
    import pickle
    yld=json.load(open('results/exp5_yield.json'))
    d0=np.array(yld["nominal"]["d"]); n0=np.array(yld["nominal"]["n"])
    dat=np.load(os.path.join(common.DATA,"process_traces.npz"))
    X=fabgan.errors_from_traces(dat["recipe_d"],dat["recipe_n"],dat["fab_d"],dat["fab_n"])
    mt=X[:,:20].mean(0); mn=X[:,20:].mean(0)   # fitted systematic bias
    # mean-only compensation: choose recipe so E[fabricated]~=nominal optimum
    d_mo,n_mo=clip(d0/(1.0+mt), n0-mn)

    st=pickle.load(open(STATE,'rb')) if os.path.exists(STATE) else {}
    t0=time.time()
    # ---- oracle hill-climb: PAIRED (common-seed) evaluation, warm-started from
    #      the best available design so the oracle is a genuine upper bound ----
    if "oracle" not in st:
        cur_d=np.array(st["cur_d"]) if "cur_d" in st else np.array(yld["fabgan"]["d"])
        cur_n=np.array(st["cur_n"]) if "cur_n" in st else np.array(yld["fabgan"]["n"])
        it=st.get("it",0); sigma=st.get("sigma",0.05)
        rng=np.random.default_rng(common.SEED+500+it)
        while it<150:
            seed=common.SEED+1000+it            # paired seed: same draws for incumbent & challenger
            base=cvar_true(cur_d,cur_n,1500,seed)
            step_d=rng.normal(0,1,20)*sigma*(THI-TLO)
            step_n=rng.normal(0,1,20)*sigma*(NHI-NLO)
            cd,cn=clip(cur_d+step_d,cur_n+step_n)
            val=cvar_true(cd,cn,1500,seed)
            if val>base:
                cur_d,cur_n=cd,cn
            it+=1
            if it%50==0: sigma*=0.90
            if time.time()-t0>32:
                st.update(cur_d=cur_d.tolist(),cur_n=cur_n.tolist(),it=it,sigma=sigma)
                pickle.dump(st,open(STATE,'wb')); print("PROGRESS oracle it",it); return
        st["oracle"]={"d":cur_d.tolist(),"n":cur_n.tolist()}
        pickle.dump(st,open(STATE,'wb')); print("oracle search done")

    # ---- final large-K common evaluation of all designs + oracle + mean-only ----
    Kbig=8000
    designs={"nominal":(d0,n0),"mean_only":(d_mo,n_mo),
             "gauss_diag":(np.array(yld["gauss_diag"]["d"]),np.array(yld["gauss_diag"]["n"])),
             "gauss_full":(np.array(yld["gauss_full"]["d"]),np.array(yld["gauss_full"]["n"])),
             "fabgan":(np.array(yld["fabgan"]["d"]),np.array(yld["fabgan"]["n"])),
             "oracle":(np.array(st["oracle"]["d"]),np.array(st["oracle"]["n"]))}
    import pickle as _pk
    res=st.get("res",{})
    for k,(d,n) in designs.items():
        if k in res: continue
        rng=np.random.default_rng(common.SEED+321)  # common draws across designs
        D,N=process.corrupt_ensemble(d,n,Kbig,rng)
        Js_list=[]
        for b in range(0,Kbig,2000):
            Jb=np.asarray(tmm_jax.notch_merit_batch(jnp.asarray(D[b:b+2000]),jnp.asarray(N[b:b+2000]),stop,pas))
            Js_list.append(Jb)
        J=np.concatenate(Js_list); Js=np.sort(J); q=max(1,int(np.ceil(0.05*Kbig)))
        res[k]={"mean":float(J.mean()),"P5":float(np.percentile(J,5)),
                "CVaR5":float(Js[:q].mean())}
        st["res"]=res; _pk.dump(st,open(STATE,'wb'))
        if time.time()-t0>32:
            print("PROGRESS eval done:",k); return
    orc=res["oracle"]["CVaR5"]
    for k in res: res[k]["regret_CVaR5"]=float(orc-res[k]["CVaR5"])
    res["_Kbig"]=Kbig; res["_bias_t_mean"]=float(mt.mean()); res["_bias_n_mean"]=float(mn.mean())
    json.dump(res,open('results/rev_oracle.json','w'),indent=1)
    os.remove(STATE)
    print(json.dumps({k:{kk:round(vv,4) for kk,vv in v.items()} for k,v in res.items() if not k.startswith('_')},indent=1))
    print("DONE")

if __name__=="__main__": main()
