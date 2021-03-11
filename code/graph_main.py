import jax.numpy as jnp
from jax.config import config; config.update('jax_enable_x64', True)
jnp.set_printoptions(precision=4)

import jax.scipy as jsc
from jax import jacfwd, jacrev, grad, jit, vmap, random, ops
import time
import numpy as np

def Tk(K, c, Shat, p):
    nc = [i for i in range(p) if i not in c]
    idx_cc = ([x for x in c for y in c], [y for x in c for y in c])
    Kcc = jnp.linalg.inv(Shat[c,:][:,c]) + K[c,:][:,nc] @ jsc.linalg.solve(K[nc,:][:,nc], K[nc,:][:,c])
    return ops.index_update(K, idx_cc, Kcc.flatten())

def T(K, C, Shat):
    p = K.shape[0]
    for c in C:
        K = Tk(K, c, Shat, p)
    return K



def itpropscaling(C, Shat):
    p = Shat.shape[0]
    tol=1e-8
    K = jnp.eye(p)
    err = 1
    while err > tol:
        Ko = K
        K = T(K, C, Shat)
        err = jnp.linalg.norm(K - Ko)
    
    return K


def chain(key, p):
    k1, k2 = random.split(key)
    idx_d = jnp.diag_indices(p)
    idx_n = ([],[])
    cliques = []
    for i in range(p-1):
        cliques.append([i, i+1])
        setidx(idx_n, i, i+1)
    
    U = jnp.zeros((p, p))
    U = ops.index_update(U, idx_d, random.normal(k1, (p,))**2)
    U = ops.index_update(U, idx_n, random.normal(k2, (p-1,)))

    return cliques, U @ U.T


def cycle(key, p):
    idx_n = ([],[])
    cliques = []
    
    for i in range(p-1):
        cliques.append([i, i+1])
        setidx(idx_n, i, i+1)

    setidx(idx_n, 0, p-1)
    cliques.append([0, p-1])

    K = ops.index_update(jnp.eye(p), idx_n, random.truncated_normal(key, -0.5, 0.5, (p,)))
    
    return cliques, (K + K.T)/2


def cycle_with_one_chord(key, p):
    k1, k2 = random.split(key)
    idx_d = jnp.diag_indices(p)
    idx_n = ([],[])
    cliques = []
    
    for i in range(p-1):
        cliques.append([i, i+1])
        setidx(idx_n, i, i+1)

    setidx(idx_n, 0, p-1)
    cliques.pop()
    cliques.append([0, p-2, p-1])

    U = jnp.zeros((p, p))
    U = ops.index_update(U, idx_d, random.normal(k1, (p,))**2)
    U = ops.index_update(U, idx_n, random.normal(k2, (p,)))
    
    return cliques, U @ U.T


def experiment(key, K, cliques0, cliques1, n):
    p = K.shape[0]
    X = random.multivariate_normal(key, jnp.zeros(p), jnp.linalg.inv(K), (n,))
    ssd = X.T @ X
    Shat = ssd/n # + np.finfo(np.float32).eps * jnp.eye(p)

    Khat0 = itpropscaling(cliques0, Shat)
    Khat1 = itpropscaling(cliques1, Shat)

    w = n*(jnp.linalg.slogdet(Khat1)[1] - jnp.linalg.slogdet(Khat0)[1])
    return jnp.exp(-w/2)

p = 50
n = 500

key = random.PRNGKey(1)
k1, k2, k3, kexp = random.split(key, 4)

chainCliques, chainK = chain(k1, p)
cycleCliques, cycleK = cycle(k2, p)
diamondCliques, _ = cycle_with_one_chord(k3, p)

k4, k5 = random.split(kexp, 2)
r1 = experiment(k4, cycleK, cycleCliques, diamondCliques, n)
r2 = experiment(k5, chainK, chainCliques, cycleCliques, n)



def foo():
    samples = 10
    t = 0
    for i in range(samples):
        start = time.perf_counter_ns()
        experiment(k4, cycleK, cycleCliques, diamondCliques, n)
        t += time.perf_counter_ns() - start

    return (t / samples) / 1e6


foo()
