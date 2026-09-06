"""R2 revision: empirical bias/variance of the pathwise empirical-CVaR
gradient estimator (R1.5/R2.7). At the nominal and FabGAN-robust designs,
compare g_K (K=256, 200 replications) with a large-sample reference
(mean of 200 gradients at K=8192). Reports relative bias of the mean
estimator, per-replication cosine similarity to the reference, and the
noise-to-signal ratio. All through the differentiable twin."""
import json, os, sys, time, pickle
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0,'.')
from src import tmm_jax, fabgan
from experiments import common
STATE="results/r2_grad_state.pkl"; STOP,PAS=tmm_jax.band_masks(common.CENTER)
gp=jax.tree.map(jnp.asarray,pickle.load(open('results/fabgan_vanilla.pkl','rb')))
def cvar_obj(dn,z):
    d,n=dn; c=fabgan.norm_recipe(d,n)
    x=fabgan.gen_forward(gp,z,jnp.tile(c,(z.shape[0],1)))
    dt,nt=fabgan.apply_errors(d,n,x)
    dt=jnp.clip(dt,0.5*tmm_jax.T_LO,2.0*tmm_jax.T_HI); nt=jnp.clip(nt,1.40,2.60)
    J=tmm_jax.notch_merit_batch(dt,nt,STOP,PAS)
    K=J.shape[0]; q=max(1,int(np.ceil(0.05*K)))
    return jnp.mean(jnp.sort(J)[:q])
g256=jax.jit(jax.grad(cvar_obj,argnums=0),static_argnums=())
def flat(g): return np.concatenate([np.asarray(g[0]),np.asarray(g[1])])
def main():
    t0=time.time(); budget=lambda: time.time()-t0>34
    inv=json.load(open('results/exp4_inverse.json'))
    designs={"nominal":(np.asarray(inv["theta_d"]),np.asarray(inv["theta_n"]))}
    # fabgan-robust design: re-derive quickly is costly; use nominal + mid-run? use nominal only if robust absent
    st=pickle.load(open(STATE,'rb')) if os.path.exists(STATE) else {"g":{}}
    R=100; KS=[64,256,2048]
    for name,(d,n) in designs.items():
        w=st["g"].setdefault(name,{str(K):[] for K in KS}); w.setdefault("r",0)
        dn=(jnp.asarray(d),jnp.asarray(n))
        while w["r"]<R:
            base=jax.random.PRNGKey(10000+w["r"]); ks=jax.random.split(base,len(KS))
            for K,kk in zip(KS,ks):
                w[str(K)].append(flat(g256(dn,jax.random.normal(kk,(K,fabgan.DIM_Z)))))
            w["r"]+=1
            if w["r"]%10==0: pickle.dump(st,open(STATE,'wb'))
            if budget(): pickle.dump(st,open(STATE,'wb')); print("PROG",name,w["r"]); return
        pickle.dump(st,open(STATE,'wb'))
    out={}
    for name,w in st["g"].items():
        G={K:np.array(w[str(K)]) for K in KS}
        ref=G[2048].mean(0); nr=np.linalg.norm(ref)
        res={"R":w["r"]}
        for K in KS:
            m=G[K].mean(0)
            cos=[float(s@ref/(np.linalg.norm(s)*nr)) for s in G[K]]
            res[f"rel_bias_K{K}"]=float(np.linalg.norm(m-ref)/nr)
            res[f"cos_mean_K{K}"]=float(np.mean(cos))
            res[f"noise_to_signal_K{K}"]=float(np.sqrt(np.trace(np.cov(G[K].T)))/nr)
        out[name]=res
    json.dump(out,open("results/r2_grad.json","w"),indent=1)
    os.remove(STATE); print(json.dumps(out,indent=1)); print("DONE")
if __name__=="__main__": main()
