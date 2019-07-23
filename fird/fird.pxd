from libc.math import exp, log

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
            newTheta = (weight[i] + theta[i] * length * lambda) / (weightSum + lamb / theta[i]) 
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
    cdef int random_state
    cdef int N = 0, M = 0, G = 0
    cdef int [:] D
    cdef double *pi, **mu, ***alpha, ***beta
    cdef double **phi, ***gamma
    cdef bint *is_outlier
    cdef double lambda_pi, lambda_alpha