import time
import numpy as np
from fird import Fird
from sklearn.metrics import homogeneity_completeness_v_measure as scoring


def generate_dataset(N, G, M, D, pi, mu, alpha):
    x, gid = np.zeros((N, M)), np.zeros(N, dtype=np.long)
    for n in range(N):
        gid[n] = np.random.choice(np.arange(G), p=pi)
        for m in range(M):
            f = np.random.rand() <= mu[gid[n]][m]
            if f:
                x[n][m] = np.random.choice(np.arange(D[m]), p=alpha[gid[n]][m])
            else:
                x[n][m] = np.random.randint(D[m])
    return x, gid


def psimplex(dim):
    res = np.random.rand(dim)
    return res / res.sum()


def sparse_psimplex(dim):
    res = psimplex(dim)
    res[np.random.randint(dim)] += 4
    return res / res.sum()


def run_fird(x, G, lambda_pi, lambda_alpha):
    start = time.time()
    model = Fird(G=G, lambda_pi=lambda_pi,
                 lambda_alpha=lambda_alpha, max_iter=100)
    resp, _ = model.fit_transform(x.astype(np.long))
    assignment = np.argmax(resp, axis=1)
    return scoring(gid, assignment), time.time() - start


N = 2000
G = 10
M = 20
D = [100] * M

pi = psimplex(G)
mu = np.random.choice([0, 1], p=[0.8, 0.2], size=(G, M)).astype(np.double)
mu = (mu + 0.1) / 1.1
alpha = np.array([[sparse_psimplex(D[m]) for m in range(M)] for _ in range(G)])

x, gid = generate_dataset(N, G, M, D, pi, mu, alpha)

print(run_fird(x, 20, 0.9, 0.9))
