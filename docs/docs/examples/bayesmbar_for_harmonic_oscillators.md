# BayesMBAR for harmonic oscillators

In this example, we will use BayesMBAR to compute the free energy differences among five states.
Each of the five states corresponds to a harmonic oscillator with a different force constant and equilibrium position.
Because the potential energy function is quadratic, the free energy differences among the five states can be computed analytically, and we will use the analytical results to validate the results from BayesMBAR.
To compute the free energy differences using BayesMBAR, we follow the following steps:

1. Draw samples from the Boltzmann distribution of each state.
2. For each sample from each state, compute its reduced potential energy in all five states. Put the reduced potential energies in a matrix, which will be used as input to BayesMBAR.
3. Use BayesMBAR to compute the free energy differences among the five states and compare the results to the analytical results.


```python
import math
import numpy as np
import jax
from sys import exit
import sys
from bayesmbar import BayesMBAR
```

Define the eqilibrium positions and force constants of the harmonic oscillators and compute the free energy differences among them using the analytical formula.


```python
M = 5 ## number of states
mu = np.linspace(0, 1, M) ## equilibrium positions
np.random.seed(0)
k = np.random.uniform(10, 30, M) ## force constants

sigma = np.sqrt(1.0 / k)
F_reference = -np.log(sigma)
```

Draw samples from the Boltzmann distribution of each state and compute the reduced potential energies of the samples in all states.


```python
n = 10000

x = [np.random.normal(mu[i], sigma[i], (n,)) for i in range(M)]
x = np.concatenate(x)

u = 0.5 * k.reshape((-1, 1)) * (x - mu.reshape((-1, 1))) ** 2
num_conf = np.array([n for i in range(M)])
```

Run BayesMBAR to compute the free energy differences among the states and compare the results with the analytical formula.


```python
mbar = BayesMBAR(
    u,
    num_conf,
    prior='uniform',
    mean=None,
    state_cv=None,
    kernel=None,
    sample_size=1000,
    warmup_steps=100,
    optimize_steps=0,
    random_seed=0,
    verbose=False
)
```

```text
Solve for the mode of the likelihood
==================================================================
                RUNNING THE NEWTON'S METHOD                     

                           * * *                                

                    Tolerance EPS = 1.00000E-12                  

At iterate    0; f= 9.68802E+00; |1/2*Newton_decrement^2|: 5.34750E-04

At iterate    1; f= 9.68749E+00; |1/2*Newton_decrement^2|: 4.78385E-08

At iterate    2; f= 9.68749E+00; |1/2*Newton_decrement^2|: 3.83538E-16

N_iter   = total number of iterations
N_func   = total number of function and gradient evaluations
F        = final function value 

             * * *     

N_iter    N_func        F
     3         5    9.687490E+00
  F = 9.687489749636 

CONVERGENCE: 1/2*Newton_decrement^2 < EPS

=====================================================
Sample from the likelihood
Sample using the NUTS sampler
```


Because we can only compute the free energy of states up to an additive constant, the free energy of states returned by BayesMBAR is shifted so that the sum of the free energies of all states is zero. We will first shift the analytical results similary and then compare the results from BayesMBAR with the analytical results.


```python
F_reference = F_reference - np.mean(F_reference)
np.set_printoptions(precision=2)

print("reference result: ", F_reference)
print("posterior mode  : ", mbar.F_mode)
print("posterior mean  : ", mbar.F_mean)
```

```text
reference result:  [-0.01  0.07  0.02 -0.01 -0.07]
posterior mode  :  [-0.02  0.07  0.02 -0.01 -0.07]
posterior mean  :  [-0.02  0.07  0.02 -0.01 -0.07]
```


BayesMBAR uses the posterior standard deviation of the free energy as an estimate of the uncertainty of the free energy.


```python
print("posterior std   : ", mbar.F_std)
```

```text
posterior std   :  [0.01 0.01 0.   0.01 0.01]
```


BayesMBAR also computes estimates and uncertainties of free energy differences between pairs of states.


```python
print("posterior mode of free energy differences: ")
print(mbar.DeltaF_mode)
```

```text
posterior mode of free energy differences: 
[[ 0.    0.08  0.04  0.01 -0.05]
 [-0.08  0.   -0.05 -0.07 -0.13]
 [-0.04  0.05  0.   -0.03 -0.09]
 [-0.01  0.07  0.03  0.   -0.06]
 [ 0.05  0.13  0.09  0.06  0.  ]]
```



```python
print("posterior mean of free energy differences: ")
print(mbar.DeltaF_mean)
```

```text
posterior mean of free energy differences: 
[[ 0.    0.08  0.04  0.01 -0.05]
 [-0.08  0.   -0.05 -0.07 -0.13]
 [-0.04  0.05  0.   -0.03 -0.09]
 [-0.01  0.07  0.03  0.   -0.06]
 [ 0.05  0.13  0.09  0.06  0.  ]]
```



```python
print("posterior std of free energy differences: ")
print(mbar.DeltaF_std)
```

```text
posterior std of free energy differences: 
[[0.   0.01 0.01 0.02 0.02]
 [0.01 0.   0.01 0.01 0.02]
 [0.01 0.01 0.   0.01 0.01]
 [0.02 0.01 0.01 0.   0.01]
 [0.02 0.02 0.01 0.01 0.  ]]
```

