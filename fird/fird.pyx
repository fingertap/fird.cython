import numpy as np
cimport numpy as np
import scipy
from libc.math cimport exp, log
from libc.stdlib cimport rand
from cpython.mem cimport PyMem_Malloc, PyMem_Free
import matplotlib.pyplot as plt

cdef inline double clip(double target, double lower, double higher) except *:
    if target < lower:
        return lower
    if target > higher:
        return higher
    return target

cdef inline double logSumExp(double [:] array, Py_ssize_t size) except *:
    cdef double maxv = -999., sumv = 0.
    cdef Py_ssize_t i
    for i in range(size):
        array[i] = clip(array[i], -500., 500.)
        if array[i] > maxv:
            maxv = array[i]
    for i in range(size):
        sumv += exp(array[i] - maxv)
    return log(sumv) + maxv

cdef inline void sparseUpdate(int length, double* theta, double [:] weight, double lamb) except *:
    cdef double normalizer, newTheta, weightSum = 0.
    cdef Py_ssize_t i, r
    for i in range(length):
        weightSum += weight[i]
    for r in range(10):
        normalizer = 0.
        # Update
        for i in range(length):
            if theta[i] < 1e-8:
                normalizer += theta[i]
                continue
            newTheta = (weight[i] + theta[i] * length * lamb) / (weightSum + lamb / theta[i]) 
            normalizer += newTheta
            theta[i] = newTheta
        for i in range(length):
            theta[i] /= normalizer

cdef inline void smoothUpdate(int length, double* theta, double [:] weight, double lamb) except *:
    cdef double sumv = 0.
    cdef Py_ssize_t i
    for i in range(length):
        theta[i] += lamb
        sumv += theta[i]
    for i in range(length):
        theta[i] /= sumv

def bincount(array, size):
    res = np.zeros(size, dtype=np.long)
    cdef Py_ssize_t i
    for i in range(size):
        res[i] = len(array[array==i])
    return res

cdef inline void check_nan(double param, name):
    if np.isnan(param):
        raise ValueError('{} contains NaN!'.format(name))

cdef class Fird:
    cdef bint trained
    cdef int random_state, max_iter
    cdef double lambda_pi, lambda_alpha, eps
    cdef int N, M, G
    cdef long [:] D
    cdef double* pi
    cdef double [:, :] mu
    cdef double ***alpha
    cdef double ***beta
    cdef double [:, :] phi
    cdef double [:, :, :] gamma
    cdef unsigned int [:] is_outlier
    def __cinit__(self, int G=50, double eps=.1, int max_iter=50,
                  double lambda_pi=0.9, double lambda_alpha=0.9):
        self.G = G
        self.max_iter = max_iter
        self.lambda_pi = lambda_pi
        self.lambda_alpha = lambda_alpha
        self.eps = eps

    cdef parse_shape(self, long [:, :] X):
        self.N, self.M = X.shape[:2]
        self.D = np.max(X, axis=0) + 1
    
    cdef check_params(self):
        cdef Py_ssize_t g, m, i
        for g in range(self.G):
            check_nan(self.pi[g], 'pi')
            for m in range(self.M):
                check_nan(self.mu[g][m], 'mu')
                for i in range(self.D[m]):
                    check_nan(self.alpha[g][m][i], 'alpha')
                    check_nan(self.beta[g][m][i], 'beta')

    def get_params(self):
        cdef Py_ssize_t g, m, i
        pi = np.array([self.pi[g] for g in range(self.G)])
        mu = np.asarray(self.mu)
        alpha = np.array([[[self.alpha[g][m][i] for i in range(self.D[m])]
                           for m in range(self.M)] for g in range(self.G)])
        beta = np.array([[[self.beta[g][m][i] for i in range(self.D[m])]
                           for m in range(self.M)] for g in range(self.G)])
        return pi, mu, alpha, beta

    def fit_transform(self, long [:, :] X):
        # Cleaning and preparing
        self.dealloc()
        self.parse_shape(X)
        self.malloc()
        self.check_params()

        # Start training

        ## Allocate intermediate varaibles
        ## NOTE: _phi is in log scale in order to prevent overflow
        cdef Py_ssize_t max_dim = np.max(self.D) + 1
        cdef double  _gamma, _gammaBar
        cdef double [:] _phi = np.zeros(self.G)
        cdef double [:] _pi = np.zeros(self.G)
        cdef double [:] _alpha = np.zeros(max_dim)
        cdef double [:] _beta = np.zeros(max_dim)

        ## EM process
        cdef double likelihood = -99999999., new_likelihood, row_sum
        cdef double weight, sumv, phi_sum, mu_sum
        cdef Py_ssize_t epoch, n, g, m, i
        for epoch in range(self.max_iter):
            # E step
            ## Calculate the evidence and update likelihood
            new_likelihood = 0.
            for n in range(self.N):
                for g in range(self.G):
                    _phi[g] = log(self.pi[g] + 1e-10)
                    if self.pi[g] < 1e-6:
                        continue
                    for m in range(self.M):
                        _gamma = self.mu[g][m] * self.alpha[g][m][X[n][m]]
                        _gammaBar = (1. - self.mu[g][m]) * self.beta[g][m][X[n][m]]
                        self.gamma[n][g][m] = _gamma / (_gamma + _gammaBar + 1e-10)
                        _phi[g] += log(_gamma + _gammaBar + 1e-10)
                # Note that by logSumExp, the array signal will get cut
                row_sum = logSumExp(_phi, self.G)
                # row_sum = scipy.special.logsumexp(_phi)
                for g in range(self.G):
                    self.phi[n][g] = exp(_phi[g] - row_sum)
                new_likelihood += row_sum
            ## The regularizers
            for g in range(self.G):
                if self.lambda_pi < 1e-8:
                    continue
                new_likelihood -= self.lambda_pi * log(self.pi[g])
                for m in range(self.M):
                    weight = self.lambda_pi * self.N / (2. * self.G * self.D[m])
                    sumv = 0.
                    for i in range(self.D[m]):
                        sumv -= log(self.alpha[g][m][i]) - log(self.beta[g][m][i])
                    new_likelihood += weight * sumv


            # M step
            ## Update the parameters using the evidence
            for g in range(self.G):
                phi_sum = 0.
                for n in range(self.N):
                    phi_sum += self.phi[n][g]
                _pi[g] = phi_sum / self.N
                if _pi[g] < 1e-8:
                    continue
                for m in range(self.M):
                    mu_sum = 0.
                    for n in range(self.N):
                        mu_sum += self.phi[n][g] * self.gamma[n][g][m]
                    self.mu[g][m] = mu_sum / phi_sum
                    for i in range(self.D[m]):
                        _alpha[i] = _beta[i] = 0.
                    for n in range(self.N):
                        _alpha[X[n][m]] += self.phi[n][g] * self.gamma[n][g][m]
                        _beta[X[n][m]] += self.phi[n][g] * (1. - self.gamma[n][g][m])
                    weight = self.lambda_alpha * self.N / (self.G * 2. * self.D[m])
                    sparseUpdate(self.D[m], self.alpha[g][m], _alpha, weight)
                    smoothUpdate(self.D[m], self.beta[g][m], _beta, weight)
            sparseUpdate(self.G, self.pi, _pi, self.lambda_pi / self.G)


            # Smoothing
            for g in range(self.G):
                self.pi[g] = (self.pi[g] + 1e-8) / (1. + self.G * 1e-8)
                for m in range(self.M):
                    self.mu[g][m] = (self.mu[g][m] + 1e-8) / (1. + 1e-8)
                    for i in range(self.D[m]):
                        self.alpha[g][m][i] = (self.alpha[g][m][i] + 1e-8) / (1. + self.D[m] * 1e-8)
                        self.beta[g][m][i] = (self.beta[g][m][i] + 1e-8) / (1. + self.D[m] * 1e-8)

            # Stop test
            if abs(new_likelihood - likelihood) < self.eps:
                break
            likelihood = new_likelihood

        # Determining the outliers (denoising)
        cdef double minv, ce, confidence
        cdef int min_pos
        cdef double [:] threshold = np.zeros(self.G)
        for g in range(self.G):
            for m in range(self.M):
                for i in range(self.D[m]):
                    ce = self.mu[g][m] * self.alpha[g][m][i] + (1. - self.mu[g][m]) * self.beta[g][m][i]
                    threshold[g] -= ce * log(ce)
        for n in range(self.N):
            self.is_outlier[n] = False
            minv = 99999999.
            min_pos = -1
            for g in range(self.G):
                confidence = 0.
                for m in range(self.M):
                    confidence -= log(self.mu[g][m] * self.alpha[g][m][X[n][m]] + (1. - self.mu[g][m]) * self.beta[g][m][X[n][m]])
                if minv > confidence:
                    minv = confidence
                    min_pos = g
            self.is_outlier[n] = minv - threshold[min_pos] > 1e-8

        # Finish training
        self.trained = True
        return np.asarray(self.phi), np.asarray(self.is_outlier, dtype=np.bool)

    def assign_labels(self, group_label, double outlier_label=0.):
        if not self.trained:
            raise ValueError("Model not trained yet!")
        assert group_label.shape[0] == self.G
        cdef Py_ssize_t n, g
        res = np.zeros(self.N)
        for n in range(self.N):
            if self.is_outlier[n]:
                res[n] = outlier_label
                continue
            for g in range(self.G):
                res[n] += self.phi[n][g] * group_label[g]
        return res

    def display(self, label, show_outliers=True):
        # label = 1 means fraud
        if not self.trained:
            raise ValueError("Model not trained yet!")
        assert len(label) == self.N
        assignments = np.argmax(self.phi, axis=1)
        f = plt.figure(figsize=(15, 7))
        width, indeces = 0.5, np.arange(self.G)
        if show_outliers:
            mask = np.asarray(self.is_outlier, dtype=np.bool)
            normal_counts = bincount(assignments[(label == 0) & ~mask], self.G)
            fraud_counts = bincount(assignments[(label == 1) & ~mask], self.G)
            filtered_normal = bincount(assignments[(label == 0) & mask], self.G)
            filtered_fraud = bincount(assignments[(label == 1) & mask], self.G)
            p1 = plt.bar(indeces, filtered_fraud, width, color='#9e9e9e')
            p2 = plt.bar(indeces, fraud_counts, width, bottom=filtered_fraud)
            p3 = plt.bar(indeces, filtered_normal, width,
                         bottom=filtered_fraud + fraud_counts,
                         color='#000000')
            p4 = plt.bar(indeces, normal_counts, width,
                         bottom=filtered_fraud + fraud_counts + filtered_normal)
            legends = ['Normal', 'Filtered normal', 'Fraud', 'Filtered fraud']
            plt.legend((p4[0], p3[0], p2[0], p1[0]), legends)
        else:
            normal_counts = bincount(assignments[label==0], self.G)
            fraud_counts = bincount(assignments[label==1], self.G)
            p1 = plt.bar(indeces, fraud_counts, width)
            p2 = plt.bar(indeces, normal_counts, width, bottom=fraud_counts)
            plt.legend((p2[0], p1[0]), ['Normal', 'Fraud'])

        ylim = bincount(assignments, self.G).max() * 1.1
        plt.title('User Counts in Each Group')
        plt.ylim([0, ylim])
        plt.xticks(
            indeces, ['group_{}'.format(x)for x in range(self.G)], rotation=70
        )
        plt.ylabel('User Count')
        return f

    cdef malloc(self):
        # TODO: memory check
        cdef Py_ssize_t g, m, n
        cdef double suma, sumb
        # cdef double [:, :] data
        # cdef double [:] sums
        # Init the parametes
        self.mu = np.full((self.G, self.M), 0.5)
        self.pi = <double*> PyMem_Malloc(self.G * sizeof(double))
        self.alpha = <double***> PyMem_Malloc(self.G * sizeof(double**))
        self.beta = <double***> PyMem_Malloc(self.G * sizeof(double**))
        # cdef double [:] pi_init = np.random.rand(self.G)
        # cdef double pi_sum = np.sum(pi_init)
        for g in range(self.G):
            self.alpha[g] = <double**> PyMem_Malloc(self.M * sizeof(double*))
            self.beta[g] = <double**> PyMem_Malloc(self.M * sizeof(double*))
            # self.pi[g] = pi_init[g] / pi_sum
            self.pi[g] = 1. / self.G
            for m in range(self.M):
                self.alpha[g][m] = <double*> PyMem_Malloc(self.D[m] * sizeof(double))
                self.beta[g][m] = <double*> PyMem_Malloc(self.D[m] * sizeof(double))
                # data = np.random.randint(1, 10, size=(2, self.D[m])).astype(np.double)
                # sums = np.sum(data, axis=1)
                suma = sumb = 0.
                for i in range(self.D[m]):
                    self.alpha[g][m][i] = rand()
                    self.beta[g][m][i] = rand()
                    suma += self.alpha[g][m][i]
                    sumb += self.beta[g][m][i]
                for i in range(self.D[m]):
                    self.alpha[g][m][i] /= suma
                    self.beta[g][m][i] /= sumb
        
        # Init the evidence
        self.phi = np.zeros((self.N, self.G))
        self.gamma = np.zeros((self.N, self.G, self.M))

        # Init other variables
        self.is_outlier = np.zeros(self.N, dtype=np.uint32)

    cdef dealloc(self):
        cdef Py_ssize_t g, m
        
        PyMem_Free(self.pi)
        self.pi = NULL

        if self.alpha is not NULL:
            for g in range(self.G):
                for m in range(self.M):
                    PyMem_Free(self.alpha[g][m])
                PyMem_Free(self.alpha[g])
            PyMem_Free(self.alpha)
            self.alpha = NULL

        if self.beta is not NULL:
            for g in range(self.G):
                for m in range(self.M):
                    PyMem_Free(self.beta[g][m])
                PyMem_Free(self.beta[g])
            PyMem_Free(self.beta)

        self.trained = False

    def __dealloc__(self):
        self.dealloc()