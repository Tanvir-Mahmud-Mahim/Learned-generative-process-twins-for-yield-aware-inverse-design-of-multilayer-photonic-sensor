"""R2 revision: additional density baselines (R1.3): Gaussian copula,
t-copula (nu=5), smoothed KDE (Silverman), and a RealNVP normalizing flow,
all fitted to the same 400 traces with no access to the hidden process.
Same protocol as rev_baselines: 30 held-out designs x 400 fresh true draws.
Resumable: rerun until DONE."""
import json, os, sys, time, pickle
import numpy as np, jax, jax.numpy as jnp, optax
from scipy import stats as sps
from scipy.stats import wasserstein_distance
sys.path.insert(0,'.')
from src import tmm_jax, process, fabgan
from experiments import common
STATE="results/r2_density_state.pkl"; STOP,PAS=tmm_jax.band_masks(common.CENTER)
def tails(J):
    Js=np.sort(J); q=max(1,int(np.ceil(0.05*len(J))))
    return float(np.percentile(J,5)), float(Js[:q].mean())
def merit(D,N):
    D=np.clip(D,0.5*tmm_jax.T_LO,2*tmm_jax.T_HI); N=np.clip(N,1.4,2.6)
    return np.asarray(tmm_jax.notch_merit_batch(jnp.asarray(D),jnp.asarray(N),STOP,PAS))

# ---------- copulas (empirical marginals, rank-based dependence) ----------
class Copula:
    def __init__(self,X,kind="gauss",nu=5):
        self.X=np.sort(X,axis=0); self.M,self.D=X.shape; self.kind=kind; self.nu=nu
        U=(sps.rankdata(X,axis=0)-0.5)/self.M
        Z=sps.norm.ppf(U) if kind=="gauss" else sps.t.ppf(U,nu)
        self.Cz=np.corrcoef(Z.T)+1e-6*np.eye(self.D)
        self.L=np.linalg.cholesky(self.Cz)
    def sample(self,rng,K):
        Z=rng.normal(size=(K,self.D))@self.L.T
        if self.kind=="t":
            g=rng.chisquare(self.nu,size=(K,1)); Z=Z/np.sqrt(g/self.nu)
            U=sps.t.cdf(Z,self.nu)
        else: U=sps.norm.cdf(Z)
        idx=np.clip((U*self.M).astype(int),0,self.M-1)
        return self.X[idx,np.arange(self.D)]
# ---------- smoothed KDE (bootstrap + Silverman bandwidth) ----------
class KDE:
    def __init__(self,X):
        self.X=X; M,D=X.shape
        self.h=(4/(D+2))**(1/(D+4))*M**(-1/(D+4))   # Silverman/Scott
        self.sd=X.std(0)
    def sample(self,rng,K):
        idx=rng.integers(0,self.X.shape[0],K)
        return self.X[idx]+rng.normal(size=(K,self.X.shape[1]))*self.sd*self.h
# ---------- RealNVP flow (unconditional over 40-D error vectors) ----------
def make_nvp(key,D=40,H=96,nl=6):
    ps=[]
    for i in range(nl):
        k1,k2,key=jax.random.split(key,3)
        d1=D//2
        def init(k,i,o): return [jax.random.normal(k,(i,o))*np.sqrt(1/i),jnp.zeros(o)]
        ks=jax.random.split(k1,4)
        ps.append({"w1":init(ks[0],d1,H),"w2":init(ks[1],H,H),"ws":init(ks[2],H,D-d1),"wt":init(ks[3],H,D-d1)})
    return ps
def nvp_fwd(ps,z):
    D=z.shape[-1]; d1=D//2; x=z
    for i,p in enumerate(ps):
        if i%2==0: a,b=x[...,:d1],x[...,d1:]
        else: b,a=x[...,:d1],x[...,d1:]
        h=jnp.tanh(a@p["w1"][0]+p["w1"][1]); h=jnp.tanh(h@p["w2"][0]+p["w2"][1])
        s=jnp.tanh(h@p["ws"][0]+p["ws"][1]); t=h@p["wt"][0]+p["wt"][1]
        b=b*jnp.exp(s)+t
        x=jnp.concatenate([a,b],-1) if i%2==0 else jnp.concatenate([b,a],-1)
    return x
def nvp_inv_logdet(ps,x):
    D=x.shape[-1]; d1=D//2; ld=0.
    for i in reversed(range(len(ps))):
        p=ps[i]
        if i%2==0: a,b=x[...,:d1],x[...,d1:]
        else: b,a=x[...,:d1],x[...,d1:]
        h=jnp.tanh(a@p["w1"][0]+p["w1"][1]); h=jnp.tanh(h@p["w2"][0]+p["w2"][1])
        s=jnp.tanh(h@p["ws"][0]+p["ws"][1]); t=h@p["wt"][0]+p["wt"][1]
        b=(b-t)*jnp.exp(-s); ld=ld-s.sum(-1)
        x=jnp.concatenate([a,b],-1) if i%2==0 else jnp.concatenate([b,a],-1)
    return x,ld
def nvp_loss(ps,x):
    z,ld=nvp_inv_logdet(ps,x)
    return -(-0.5*(z**2).sum(-1)-0.5*x.shape[-1]*np.log(2*np.pi)+ld).mean()

def main():
    t0=time.time(); budget=lambda: time.time()-t0>34
    dat=np.load(os.path.join(common.DATA,"process_traces.npz"))
    rd,rn,fd,fn=dat["recipe_d"],dat["recipe_n"],dat["fab_d"],dat["fab_n"]
    X=fabgan.errors_from_traces(rd,rn,fd,fn)
    mu,sd=X.mean(0),X.std(0)+1e-8; Xn=(X-mu)/sd
    bench=np.load(os.path.join(common.DATA,"benchmark_designs.npz"))
    used=set(map(tuple,np.round(rd[::common.RUNS_PER_RECIPE],9)))
    held=[i for i in range(bench["d_um"].shape[0]) if tuple(np.round(bench["d_um"][i],9)) not in used][:30]
    st=pickle.load(open(STATE,'rb')) if os.path.exists(STATE) else {"flow":None,"acc":{},"i":0}
    # train flow once (resumable)
    if st["flow"] is None or st["flow"].get("t",0)<2500:
        f=st["flow"] or {"ps":make_nvp(jax.random.PRNGKey(3)),"t":0}
        ps=jax.tree.map(jnp.asarray,f["ps"]); opt=optax.adam(1e-3); os_=opt.init(ps) if f.get("os") is None else f["os"]
        os_=jax.tree.map(jnp.asarray,os_)
        step=jax.jit(lambda ps,os_,x: (lambda l,g: (optax.apply_updates(ps,opt.update(g,os_)[0]),opt.update(g,os_)[1],l))(*jax.value_and_grad(nvp_loss)(ps,x)))
        rng=np.random.default_rng(f["t"])
        while f["t"]<2500:
            xb=jnp.asarray(Xn[rng.integers(0,Xn.shape[0],128)])
            ps,os_,l=step(ps,os_,xb); f["t"]+=1
            if budget():
                f["ps"]=jax.tree.map(np.asarray,ps); f["os"]=jax.tree.map(np.asarray,os_)
                st["flow"]=f; pickle.dump(st,open(STATE,'wb')); print("PROG flow",f["t"],float(l)); return
        f["ps"]=jax.tree.map(np.asarray,ps); f["os"]=None; st["flow"]=f
        pickle.dump(st,open(STATE,'wb')); print("flow trained")
    ps=jax.tree.map(jnp.asarray,st["flow"]["ps"])
    models={"copula_g":Copula(X,"gauss"),"copula_t":Copula(X,"t",5),"kde":KDE(X)}
    names=["copula_g","copula_t","kde","flow"]
    acc=st["acc"]
    for m in names: acc.setdefault(m,{"p5":[],"cv":[],"w":[]})
    for i in range(st["i"],len(held)):
        hi=held[i]; d,n=bench["d_um"][hi],bench["n0"][hi]
        rng=np.random.default_rng(common.SEED+900+hi)
        Dt,Nt=process.corrupt_ensemble(d,n,400,rng); Jt=merit(Dt,Nt); p5t,cvt=tails(Jt)
        for m in names:
            if m=="flow":
                z=jax.random.normal(jax.random.PRNGKey(common.SEED+900+hi),(400,40))
                xs=np.asarray(nvp_fwd(ps,z))*sd+mu
            else: xs=models[m].sample(rng,400)
            Jm=merit(d*(1+xs[:,:20]), n+xs[:,20:]); p5m,cvm=tails(Jm)
            acc[m]["p5"].append(abs(p5t-p5m)); acc[m]["cv"].append(abs(cvt-cvm)); acc[m]["w"].append(wasserstein_distance(Jt,Jm))
        st["i"]=i+1; pickle.dump(st,open(STATE,'wb'))
        if budget(): print("PROG design",i+1,"/",len(held)); return
    out={m:{"P5_abs_err_mean":float(np.mean(acc[m]["p5"])),"CVaR5_abs_err_mean":float(np.mean(acc[m]["cv"])),
            "induced_perf_W1_mean":float(np.mean(acc[m]["w"]))} for m in names}
    json.dump(out,open("results/r2_density.json","w"),indent=1)
    os.remove(STATE); print(json.dumps(out,indent=1)); print("DONE")
if __name__=="__main__": main()
