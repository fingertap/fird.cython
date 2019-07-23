import numpy as np
from libc.math cimport exp, log
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef inline double clip(double target, double lower, double higher):
    if target < lower:
        return lower
    if target > higher:
        return higher
    return target

cdef inline double logSumExp(double* array, Py_ssize_t size):
    cdef double minv = 99999., sumv = 0.
    cdef Py_ssize_t i
    for i in range(size):
        array[i] = clip(array[i], -500., 500.)
        if array[i] < minv:
            minv = array[i]
    for i in range(size):
        sumv += exp(array[i] - minv)
    return log(sumv) + minv

cdef inline void sparseUpdate(int length, double* theta, double* weight, double lamb):
    cdef double normalizer, newTheta, weightSum = 0.
    cdef Py_ssize_t i, r
    for i in range(length):
        weightSum += weight[i]
    for r in range(10):
        normalizer = 0.
        # Update
        for i in range(length):
            newTheta = (weight[i] + theta[i] * length * lamb) / (weightSum + lamb / theta[i]) 
            normalizer += newTheta
            theta[i] = newTheta
        for i in range(length):
            theta[i] /= normalizer

cdef inline void smoothUpdate(int length, double* theta, double* weight, double lamb):
    cdef double sumv = 0.
    cdef Py_ssize_t i
    for i in range(length):
        theta[i] += lamb
        sumv += theta[i]
    for i in range(length):
        theta[i] /= sumv

cdef class Fird:
    cdef bint trained
    cdef int random_state, max_iter
    cdef double lambda_pi, lambda_alpha, eps
    cdef int N, M, G
    cdef int [:] D
    cdef double *pi
    cdef double **mu
    cdef double ***alpha
    cdef double ***beta
    cdef double **phi
    cdef double ***gamma
    cdef bint *is_outlier
    def __cinit__(self, int G=50, double eps=.1, int max_iter=50,
                  double lambda_pi=0.9, double lambda_alpha=0.9):
        self.G = G
        self.max_iter = max_iter
        self.lambda_pi = lambda_pi
        self.lambda_alpha = lambda_alpha
        self.eps = eps

    cdef parse_shape(self, int [:, :] X):
        self.N, self.M = X.shape[:2]
        self.D = np.max(X, axis=0)

    def fit(self, int [:, :] X):
        # Cleaning and preparing
        self.dealloc()
        self.parse_shape(X)
        self.malloc()
        self.initialize()  # initialize values for params

        # Start training

        ## Allocate intermediate varaibles
        ## NOTE: _phi is in log scale in order to prevent overflow
        cdef double  _gamma, _gammaBar
        cdef double *_phi
        cdef double *_pi
        cdef double *_alpha
        cdef double *_beta
        cdef Py_ssize_t max_dim = np.max(self.D)
        _phi = <double*> PyMem_Malloc(self.G * sizeof(double))
        _pi = <double*> PyMem_Malloc(self.G * sizeof(double))
        _alpha = <double*> PyMem_Malloc(max_dim * sizeof(double))
        _beta = <double*> PyMem_Malloc(max_dim * sizeof(double))

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
                    _phi[g] = log(self.pi[g])
                    for m in range(self.M):
                        _gamma = self.mu[g][m] * self.alpha[g][m][X[n][m]]
                        _gammaBar = (1. - self.mu[g][m]) * self.beta[g][m][X[n][m]]
                        _phi[g] += log(_gamma + _gammaBar)
                        self.gamma[n][g][m] = _gamma / (_gamma + _gammaBar)
                # Note that by logSumExp, the array signal will get cut
                row_sum = logSumExp(_phi, self.G)
                for g in range(self.G):
                    self.phi[n][g] = exp(_phi[g] - row_sum)
                new_likelihood += row_sum
            ## The regularizers
            for g in range(self.G):
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
                for m in range(self.M):
                    mu_sum = 0.
                    for n in range(self.N):
                        muSum += self.phi[n][g] * self.gamma[n][g][m]
                    self.mu[g][m] = muSum / phi_sum
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

        # Deallocate intermediate variables
        PyMem_Free(_phi)
        PyMem_Free(_alpha)
        PyMem_Free(_beta)

        # Determining the outliers (denoising)
        cdef double minv, ce, confidence
        cdef int min_pos
        cdef double* threshold = <double*> PyMem_Malloc(self.G * sizeof(double))
        for g in range(self.G):
            threshold[g] = 0.
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
        PyMem_Free(threshold)

        # Finish training
        self.trained = True

    cdef malloc(self):
        # TODO: memory check
        cdef Py_ssize_t g, m, n
        # Init the parametes
        self.pi = <double*> PyMem_Malloc(self.G * sizeof(double))
        self.mu = <double**> PyMem_Malloc(self.G * sizeof(double*))
        self.alpha = <double***> PyMem_Malloc(self.G * sizeof(double**))
        self.beta = <double***> PyMem_Malloc(self.G * sizeof(double**))
        for g in range(self.G):
            self.mu[g] = <double*> PyMem_Malloc(self.M * sizeof(double))
            self.alpha[g] = <double**> PyMem_Malloc(self.M * sizeof(double*))
            self.beta[g] = <double**> PyMem_Malloc(self.M * sizeof(double*))
            for m in range(self.M):
                self.alpha[g][m] = <double*> PyMem_Malloc(self.D[m] * sizeof(double))
                self.beta[g][m] = <double*> PyMem_Malloc(self.D[m] * sizeof(double))
        
        # Init the evidence
        self.phi = <double**> PyMem_Malloc(self.N * sizeof(double*))
        self.gamma = <double***> PyMem_Malloc(self.N * sizeof(double**))
        for n in range(self.N):
            self.phi[n] = <double*> PyMem_Malloc(self.G * sizeof(double))
            self.gamma[n] = <double**> PyMem_Malloc(self.G * sizeof(double*))
            for g in range(self.G):
                self.gamma[n][g] = <double*> PyMem_Malloc(self.M * sizeof(double))

        # Init other variables
        self.is_outlier = <bint*> PyMem_Malloc(self.N * sizeof(bint))

    cdef initialize(self):
        cdef Py_ssize_t n, g, m, i
        cdef double [:, :] data
        cdef double [:] sums
        for n in range(self.N):
            self.is_outlier[n] = False
        for g in range(self.G):
            self.pi[g] = 1. / self.G
            for m in range(self.M):
                self.mu[g][m] = 0.5
                data = np.random.rand(2, self.D[m])
                sums = np.sum(data, axis=1)
                for i in range(self.D[m]):
                    self.alpha[g][m][i] = data[0][i] / sums[0]
                    self.beta[g][m][i] = data[1][i] / sums[1]

    cdef dealloc(self):
        cdef Py_ssize_t g, m, n

        if self.pi is not NULL:
            PyMem_Free(self.pi)
            self.pi = NULL

        if self.mu is not NULL:
            for g in range(self.G):
                PyMem_Free(self.mu[g])
            PyMem_Free(self.mu)
            self.mu = NULL

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

        if self.phi is not NULL:
            for n in range(self.N):
                PyMem_Free(self.phi[n])
            PyMem_Free(self.phi)
            self.phi = NULL

        if self.gamma is not NULL:
            for n in range(self.N):
                for g in range(self.G):
                    PyMem_Free(self.gamma[n][g])
                PyMem_Free(self.gamma[n])
            PyMem_Free(self.gamma)
            self.gamma = NULL
            
        if self.is_outlier is not NULL:
            PyMem_Free(self.is_outlier)
            self.is_outlier = NULL

        self.trained = False

    def __dealloc__(self):
        self.dealloc()