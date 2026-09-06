"""R2 revision: second design objective (R1.7/R2.9): 600-nm band-pass filter
on the same 20-layer platform. Nominal probe-seeded adjoint design, then
robustification under the existing FabGAN twin vs the full Gaussian twin,
evaluated on the hidden true process (paired draws). Resumable."""
import json, os, sys, time, pickle
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0,'.')
from src import tmm_jax, process, fabgan, adjoint
from experiments import common
STATE="results/r2_bandpass_state.pkl"
LAM=np.asarray(tmm_jax.LAM_UM)
PASS=((LAM>=0.585)&(LAM<=0.615))
STOPB=(((LAM>=0.40)&(LAM<=0.555))|((LAM>=0.645)&(LAM<=0.80)))
PASSj,STOPj=jnp.asarray(PASS),jnp.asarray(STOPB)
def bp_merit_batch(D,N):
    T=tmm_jax.transmittance_batch(D,N)
    return 0.5*T[:,PASSj].mean(1)+0.5*(1.0-T[:,STOPj].mean(1))
def bp_merit(d,n):
    T=tmm_jax.transmittance_jit(d,n)
    return 0.5*T[PASSj].mean()+0.5*(1.0-T[STOPj].mean())
def stats(J):
    Js=np.sort(J); q=max(1,int(np.ceil(0.05*len(J))))
    return {"mean":float(J.mean()),"P5":float(np.percentile(J,5)),"CVaR5":float(Js[:q].mean())}
def main():
    t0=time.time(); budget=lambda: time.time()-t0>34
    st=pickle.load(open(STATE,'rb')) if os.path.exists(STATE) else {}
    if "d0" not in st:
        # nominal design: 200 probes + 4 x 40 adjoint refinements (same budget as notch)
        import optax
        rng=np.random.default_rng(0)
        d,n=adjoint.random_designs(rng,200)
        J=np.asarray(bp_merit_batch(jnp.asarray(d),jnp.asarray(n)))
        order=np.argsort(J)[::-1]
        vg=jax.jit(jax.value_and_grad(bp_merit,argnums=(0,1)))
        best=(None,None,-1.)
        for s in order[:4]:
            dd,nn=jnp.asarray(d[s]),jnp.asarray(n[s])
            opt=optax.adam(4e-3); stt=opt.init((dd,nn))
            for _ in range(40):
                Jc,g=vg(dd,nn)
                up,stt=opt.update(jax.tree.map(lambda x:-x,g),stt)
                dd,nn=optax.apply_updates((dd,nn),up); dd,nn=tmm_jax.clip_design(dd,nn)
                if float(Jc)>best[2]: best=(np.asarray(dd),np.asarray(nn),float(Jc))
        st["d0"],st["n0"],st["J0"]=best
        pickle.dump(st,open(STATE,'wb')); print("PROG nominal J*=%.4f"%best[2]); return
    gp=jax.tree.map(jnp.asarray,pickle.load(open('results/fabgan_vanilla.pkl','rb')))
    dat=np.load(os.path.join(common.DATA,"process_traces.npz"))
    X=fabgan.errors_from_traces(dat["recipe_d"],dat["recipe_n"],dat["fab_d"],dat["fab_n"])
    mu=X.mean(0); L=np.linalg.cholesky(np.cov(X.T)+1e-9*np.eye(X.shape[1]))
    def gan_obj():
        def obj(dn,z):
            d,n=dn; c=fabgan.norm_recipe(d,n)
            x=fabgan.gen_forward(gp,z,jnp.tile(c,(256,1)))
            dt,nt=fabgan.apply_errors(d,n,x)
            dt=jnp.clip(dt,0.5*tmm_jax.T_LO,2.0*tmm_jax.T_HI); nt=jnp.clip(nt,1.40,2.60)
            J=bp_merit_batch(dt,nt); K=J.shape[0]; q=max(1,int(np.ceil(0.05*K)))
            return jnp.mean(jnp.sort(J)[:q])
        return jax.jit(jax.value_and_grad(obj,argnums=0))
    def gauss_obj():
        muj,Lj=jnp.asarray(mu),jnp.asarray(L)
        def obj(dn,z):
            d,n=dn; x=muj+z@Lj.T
            dt=d*(1.0+x[:,:20]); nt=n+x[:,20:]
            dt=jnp.clip(dt,0.5*tmm_jax.T_LO,2.0*tmm_jax.T_HI); nt=jnp.clip(nt,1.40,2.60)
            J=bp_merit_batch(dt,nt); K=J.shape[0]; q=max(1,int(np.ceil(0.05*K)))
            return jnp.mean(jnp.sort(J)[:q])
        return jax.jit(jax.value_and_grad(obj,argnums=0))
    for key,mk,zd in [("fabgan",gan_obj,fabgan.DIM_Z),("gauss_full",gauss_obj,40)]:
        if f"d_{key}" in st: continue
        dd,nn,out=adjoint.robustify(st["d0"],st["n0"],mk(),zd,256,steps=200,seed=0,
            resume=st.get(f"r_{key}"),max_seconds=max(5,32-(time.time()-t0)))
        if dd is None:
            st[f"r_{key}"]=out; pickle.dump(st,open(STATE,'wb')); print("PROG rob",key); return
        st[f"d_{key}"],st[f"n_{key}"]=dd,nn; st.pop(f"r_{key}",None)
        pickle.dump(st,open(STATE,'wb'))
        if budget(): print("PROG rob done",key); return
    ev=st.setdefault("ev",{})
    for key,(dd,nn) in {"nominal":(st["d0"],st["n0"]),"gauss_full":(st["d_gauss_full"],st["n_gauss_full"]),
                         "fabgan":(st["d_fabgan"],st["n_fabgan"])}.items():
        if key in ev: continue
        rng=np.random.default_rng(888)
        D,Nn=process.corrupt_ensemble(dd,nn,1500,rng)
        J=np.asarray(bp_merit_batch(jnp.asarray(D),jnp.asarray(Nn)))
        ev[key]=stats(J); pickle.dump(st,open(STATE,'wb'))
        if budget(): print("PROG eval",key); return
    out={"J_nominal":st["J0"],**{k:ev[k] for k in ev}}
    json.dump(out,open("results/r2_bandpass.json","w"),indent=1)
    os.remove(STATE); print(json.dumps(out,indent=1)); print("DONE")
if __name__=="__main__": main()
