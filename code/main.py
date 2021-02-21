


# import jax.numpy as jnp
# from jax import grad, jit, random, ops
# import time

# def loglikelihood(K, Khat):
#     return jnp.linalg.slogdet(K)[1] - jnp.trace(jnp.linalg.solve(Khat, K))

# def sampleK(key, d):
#     idx_u = jnp.triu_indices(d)
#     idx_d = jnp.diag_indices(d)

#     L = random.normal(key, (d,d))
#     L = ops.index_update(L, idx_u, 0.0)
#     L = ops.index_update(L, idx_d, random.normal(key, (d,))**2)

#     return jnp.matmul(L, L.T)

# d = 100
# key = random.PRNGKey(0)
# k1,k2 = random.split(key, 2)

# K = sampleK(k1, d)
# Khat = sampleK(k2, d)

# ddl = jit(grad(loglikelihood))
# samples = 100

# def foo(samples):        
#     start = time.time()
#     for i in range(samples):
#         a = ddl(K, Khat)

#     d = time.time() - start
#     return a, d / samples


# foo(samples)

import jax.numpy as jnp
from jax.config import config; config.update('jax_enable_x64', True)

import jax.scipy as jsc
from jax import jacfwd, jacrev, grad, jit, vmap, random, ops
import time
import numpy as np

def chol_inv(L):
    d = L.shape[0]
    return jsc.linalg.solve_triangular(L, jnp.eye(d), lower=True)

def chol_solve(L, b):
    x = jsc.linalg.solve_triangular(L, b, lower=True)
    return jsc.linalg.solve_triangular(L.T, x, lower=False)

def chol_materialize(L):
    return L @ L.T

def chol_logdet(L):
    d = L.shape[0]
    idx_d = jnp.diag_indices(d)
    return 2.0*jnp.log(L[idx_d]).sum()

def chol_sample(key, d):
    idx_u = jnp.triu_indices(d)
    idx_d = jnp.diag_indices(d)

    L = random.normal(key, (d,d), dtype=jnp.float64)
    L = ops.index_update(L, idx_u, 0.0)
    L = ops.index_update(L, idx_d, random.normal(key, (d,))**2)
   
    return L

def vec_to_tril(vec):
    d = int((np.sqrt(1 + 8*vec.shape[0])-1)/2)
    idx_l = jnp.tril_indices(d)
    L = jnp.zeros((d,d), dtype=jnp.float64)
    return ops.index_update(L, idx_l, vec)

def tril_to_vec(L):
    d = L.shape[0]
    idx_l = jnp.tril_indices(d)
    return L[idx_l]


def T(K, c, nc, Shat):
    idx_cc = ([x for x in c for y in c], [y for x in c for y in c])
    Kcc = jnp.linalg.inv(Shat[c,:][:,c]) + K[c,:][:,nc] * jsc.linalg.solve(K[nc,:][:,nc], K[nc,:][:,c])
    return ops.index_update(K, idx_cc, Kcc.flatten())

def contrainedMLE(p, a, b):
    ca = [i for i in range(p) if i != b]
    cb = [i for i in range(p) if i != a]
    return lambda Shat: T(T(jnp.eye(p), ca, [b], Shat), cb, [a], Shat)


def loglikelihood_flat(K_flat, Khat_flat):
    K = vec_to_tril(K_flat)
    Khat = vec_to_tril(Khat_flat)
    return loglikelihood2(K,Khat)

def loglikelihood2(K, Khat):
    return n/2*(chol_logdet(K) - jnp.trace(chol_solve(Khat, K) @ K.T))

def loglikelihood(K, Shat_full):
    return n/2*(chol_logdet(K) - jnp.trace(Shat_full @ K @ K.T))

def w(K, Khat, Shat_full):
    l = loglikelihood(K, Shat_full)
    lhat = n/2*(chol_logdet(Khat) - K.shape[0])
    return 2*(lhat - l)

def hessian(f):
    return jacfwd(jacrev(f))


n = 1000
d = 100
key = random.PRNGKey(1)
kk, kx, kt = random.split(key, 3)

S = chol_sample(kk, d)
K = chol_inv(S)

X = S @ random.multivariate_normal(kx, jnp.zeros(d), jnp.eye(d), (n,)).T
XX = X @ X.T
Shat_full = XX/n + np.finfo(np.float32).eps * jnp.eye(d)

Shat = jsc.linalg.cholesky(Shat_full, lower=True)
Khat = chol_inv(Shat)

Khat0_full = contrainedMLE(d, 0, 1)(Shat_full)
Khat0 = jsc.linalg.cholesky(Khat0_full, lower=True)

Khat0_flat = tril_to_vec(Khat0)
Khat_flat = tril_to_vec(Khat)

w(Khat0, Khat, Shat_full)



dl = jit(grad(loglikelihood_flat))
ddl = jit(hessian(loglikelihood_flat))

dl2 = jit(grad(loglikelihood2))
ddl2 = jit(hessian(loglikelihood2))


def foo():
    samples = 10
    t = 0
    for i in range(samples):
        start = time.perf_counter_ns()
        #dl2(K, Khat)
        ddl(K_flat, Khat0_flat)
        t += time.perf_counter_ns() - start

    return (t / samples) / 1e6

foo()