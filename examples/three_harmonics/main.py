import math
import numpy as np
import jax
from sys import exit
import sys
#sys.path.append("/cluster/home/xding07/dinglab/xding07/projects_on_github/BayesMBAR")
from bayesmbar import BayesMBAR

M = 3 ## number of states
mu = np.array([0.0, 1.0, 2.0])
k = np.array([16.0, 25.0, 36.0])
sigma = np.sqrt(1.0 / k)

F_true = np.array([-np.log(sigma[i] / sigma[0]) for i in range(1, M)])

n = 10000

x = [np.random.normal(mu[i], sigma[i], (n,)) for i in range(M)]
x = np.concatenate(x)

u = 0.5 * k.reshape((-1, 1)) * (x - mu.reshape((-1, 1))) ** 2
num_conf = np.array([n for i in range(M)])


mbar = BayesMBAR(
    u,
    num_conf,
    prior='uniform',
    mean=None,
    state_cv=None,
    kernel=None,
    sample_size=2000,
    warmup_steps=200,
    optimize_steps=0,
    random_seed=0
)
