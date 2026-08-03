"""Design-gradient validation (reviewer 1 #7): the paper optimizes THROUGH the
twin, so the recipe-gradient dG/dc must be correct, not merely the samples.
We compare the FabGAN pathwise directional derivative of the mean merit against
common-random-number central finite differences of the TRUE simulator, at
held-out recipes and along random directions.
"""
import json, os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0,'.')
from src import tmm_jax, process, fabgan
from experiments import common
import pickle
STOP,PAS=tmm_jax.band_masks(common.CENTER)
TLO,THI,NLO,NHI=tmm_jax.T_LO,tmm_jax.T_HI,1.6,2.4
gp=jax.tree.map(jnp.asarray,pickle.load(open('results/fabgan_vanilla.pkl','rb')))
Z=jax.random.normal(jax.random.PRNGKey(0),(512,fabgan.DIM_Z))   # fixed latent batch

def twin_meanmerit(d,n):
    c=fabgan.norm_recipe(d,n)
    x=fabgan.gen_forward(gp,Z,jnp.tile(c,(Z.shape[0],1)))
    dt,nt=fabgan.apply_errors(d,n,x)
    dt=jnp.clip(dt,0.5*TLO,2*THI); nt=jnp.clip(nt,1.4,2.6)
    return jnp.mean(tmm_jax.notch_merit_batch(dt,nt,STOP,PAS))
grad_twin=jax.jit(jax.grad(lambda dn: twin_meanmerit(dn[0],dn[1])))

def true_meanmerit(d,n,seed,K=1500):
    rng=np.random.default_rng(seed)
    D,N=process.corrupt_ensemble(np.asarray(d),np.asarray(n),K,rng)
    return float(np.mean(np.asarray(tmm_jax.notch_merit_batch(jnp.asarray(D),jnp.asarray(N),STOP,PAS))))

bench=np.load(os.path.join(common.DATA,"benchmark_designs.npz"))
dat=np.load(os.path.join(common.DATA,"process_traces.npz"))
used=set(map(tuple,np.round(dat["recipe_d"][::common.RUNS_PER_RECIPE],9)))
held=[i for i in range(bench["d_um"].shape[0]) if tuple(np.round(bench["d_um"][i],9)) not in used][:14]
rng=np.random.default_rng(common.SEED+11)
h=0.02
tw,tr=[],[]
for i in held:
    d=bench["d_um"][i]; n=bench["n0"][i]
    g=grad_twin((jnp.asarray(d),jnp.asarray(n)))
    gd=np.asarray(g[0]); gn=np.asarray(g[1])
    v_d=rng.normal(0,1,20)*(THI-TLO); v_n=rng.normal(0,1,20)*(NHI-NLO)
    nrm=np.sqrt((v_d**2).sum()+(v_n**2).sum()); v_d/=nrm; v_n/=nrm
    dir_twin=float((gd*v_d).sum()+(gn*v_n).sum())
    seed=common.SEED+5000+i                               # common noise for +/- steps
    fp=true_meanmerit(d+h*v_d,n+h*v_n,seed); fm=true_meanmerit(d-h*v_d,n-h*v_n,seed)
    dir_true=(fp-fm)/(2*h)
    tw.append(dir_twin); tr.append(dir_true)
tw=np.array(tw); tr=np.array(tr)
pear=float(np.corrcoef(tw,tr)[0,1])
sign=float(np.mean(np.sign(tw)==np.sign(tr)))
relerr=float(np.median(np.abs(tw-tr)/(np.abs(tr)+1e-9)))
out={"n_pairs":len(held),"pearson_r":pear,"sign_agreement":sign,
     "median_rel_err":relerr,"twin_dirderiv":tw.tolist(),"true_dirderiv":tr.tolist()}
json.dump(out,open('results/rev_grad.json','w'),indent=1)
print("pairs",len(held),"pearson r=%.3f"%pear,"sign-agree=%.2f"%sign,"median rel err=%.2f"%relerr)
