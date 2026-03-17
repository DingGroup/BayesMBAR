import jax
import jax.numpy as jnp
import pickle
from bayesmbar import BayesBAR

jax.config.update("jax_enable_x64", True)

with open("/home/xding1/downloads/log_probs_state_ligand.pkl", "rb") as f:
    log_probs_state_ligand = pickle.load(f)

logqs = jnp.concatenate(
    [log_probs_state_ligand["log_qs"]["flow"], log_probs_state_ligand["log_qs"]["md"]]
)

logps = jnp.concatenate(
    [log_probs_state_ligand["log_ps"]["flow"], log_probs_state_ligand["log_ps"]["md"]]
)

uq = -logqs
up = -logps

u = jnp.stack([uq, up], axis=0)

nq = len(log_probs_state_ligand["log_qs"]["flow"])
np = len(log_probs_state_ligand["log_qs"]["md"])
num_conf = jnp.array([nq, np])
bar = BayesBAR(u,num_conf, method = 'L-BFGS', verbose = True)

with open("/home/xding1/downloads/log_probs_state_complex.pkl", "rb") as f:
    log_probs_state_complex = pickle.load(f)

with open("/home/xding1/downloads/bar_results.pkl", "rb") as f:
    bar_results = pickle.load(f)
