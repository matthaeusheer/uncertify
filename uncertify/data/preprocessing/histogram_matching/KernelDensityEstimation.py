import numpy as np
import scipy.linalg

from uncertify.data.preprocessing import binomial_functions


## RULE OF THUMB METHODS FOR ESTIMATING KERNEL WIDTH ##

# 1. Scott's rule
def ScottWidthEstimate(n, d):
    """ 
    this function implements the Scott's kernel width estimation
    for univariate Gaussian kernels (including product of them).
    n: is the number of data points
    d: is the dimension of each data point
    """
    return n ** (-1. / (d + 4))


# 2. Silverman's rule
def SilvermanWidthEstimate(n, d):
    """
    this function implements the Silverman's kernel width estimation
    for univariate Gaussian kernels (including product of them).
    n: is the number of data points
    d: is the dimension of each data point
    """
    return (n * (d + 2) / 4.) ** (-1. / (d + 4))


## DISCRETE KERNELS ##

# 1. The Binomial kernel and its multivariate extension have been taken from:
# http://www.stat.cmu.edu/~cshalizi/402/lectures/06-density/lecture-06.pdf
# They have some problems. The binomial kernel produces undesirable effects
# such as not yielding a smooth bump depending on the h and c.
def BinomialKernel(x, y, h, c):
    """
    this function implements the binomial kernel (1D)

    K(x,y|c,h) = \dbinom{c}{|x-y|}h^{|x-y|}(1-h)^{c-|x-y|}

    x, y are random discrete variables in [0,c]
    x is the sample the kernel is built around.
    y is the point at which kernel is evaluated.
    y can also be an np.array. In this case the output will be
      an array and each element will be the kernel evaluation at the
      corresponding y value.
    h is the kernel width and h \in (0,1)
    IMPORTANT NOTE:
    The Binomial kernel and its multivariate extension have been taken from:
    http://www.stat.cmu.edu/~cshalizi/402/lectures/06-density/lecture-06.pdf
    They have some problems. The binomial kernel produces undesirable effects
    such as not yielding a smooth bump depending on the h and c.
    """
    if np.ndim(y) == 0:
        return binomial_functions.binomial_value(np.abs(x - y), c, h)
    else:
        res = np.zeros([np.size(y), 1])
        for n in range(np.size(y)):
            res[n] = binomial_functions.binomial_value(np.abs(x - y[n]), c, h)
        return res


def BinomialProductKernel(x, y, h, c):
    """
    this function implements the binomial product kernel:

    K(x,y|c,h) = \prod_i\dbinom{c}{|x_i-y_i|}h_i^{|x_i-y_i|}(1-h_i)^{c-|x_i-y_i|}

    x,y D dimensional discrete random vectors.
    x is the sample vector the kernel is built around.
    y is the test vector at which kernel is evaluated.
    y can also be a np.array of random vectors. In this case the result
      will be an array and each element will kernel evaluated at the
      corresponding y vector.
      The assumed structure of y is Samples X Dimensions
      The assumed structure of x is 1 X Dimensions
    c is the maximum value a random variable can have. implicitly we
      assume that the minimum value a random variable can have is 0.
    h is the kernel width vector with h_i \in (0,1) \forall i
    this structure assumes isotropic kernel.
    IMPORTANT NOTE:
    The Binomial kernel and its multivariate extension have been taken from:
    http://www.stat.cmu.edu/~cshalizi/402/lectures/06-density/lecture-06.pdf
    They have some problems. The binomial kernel produces undesirable effects
    such as not yielding a smooth bump depending on the h and c.
    """
    value = np.ones([y.shape[0], 1])
    for n in range(y.shape[1]):
        value = value * BinomialKernel(x[:, n].reshape([x.shape[0], 1]),
                                       y[:, n].reshape([y.shape[0], 1]),
                                       h[n], c)
    return value


## CONTINUOUS KERNELS ##

# 2. The Gaussian kernel and its multivariate extension
def GaussianKernel(x, y, h, c=0):
    """
    this function implements the Gaussian kernel (1D)

    K(x,y|h) = \frac{1}{h\sqrt{2\pi}}\exp{-0.5*((x-y)/h)^2}
    
    x,y are random variables.
    x is the sample the kernel is built around.
    y is the point at which kernel is evaluated.
    y can also be an np.array. In this case the output will be
      an array and each element will be the kernel evaluation at the 
      corresponding y value.
    h is the standard deviation or the width of the kernel.
    c is a null variable that is only here to have a common
    representation of the kernels.
    """

    value = np.exp(-0.5 * (x - y) ** 2. / h) / (np.sqrt(h) * np.sqrt(2 * np.pi))
    return value


def GaussianProductKernel(x, y, h, c=0):
    """ 
    this function implements the Gaussian product kernel.

    $K(x,y|h) = \prod_i\frac{1}{h_i\sqrt{2\pi}}\exp{-0.5*((x_i-y_i)/h_i)^2}$

    x,y are D dimensional random vectors.
    x is the sample vector the kernel is built around.
    y is the test vector at which kernel is evaluated.
    y can also be a np.array of random vectors. In this case the result
      will be an array and each element will kernel evaluated at the 
      corresponding y vector.
      The assumed structure of y is Samples X Dimensions
      The assumed structure of x is 1 X Dimensions
    h is the kernel width vector.
    c is a null variable.
    """
    value = np.ones([y.shape[0], 1])
    for n in range(y.shape[1]):
        value = value * GaussianKernel(x[:, n].reshape([x.shape[0], 1]),
                                       y[:, n].reshape([y.shape[0], 1]),
                                       h[n])
    return value


def GaussianProductKernelMarginal(x, y, h, j, c=0):
    """
    this function implements the Gaussian product kernel marginal.

    for x, y and h see the definition of GaussianProductKernel.

    j is a vector of dimensions. The marginal distribution will be on the
        elements corresponding to these dimensions.
        The assumed structure of j is an array. 
    """
    x_new = x[:, j].reshape([x.shape[0], j.shape[0]])
    h_new = h[j]
    if y.shape[1] > j.shape[0]:
        y_new = y[:, j].reshape([y.shape[0], j.shape[0]])
        return GaussianProductKernel(x_new, y_new, h_new, c)
    else:
        return GaussianProductKernel(x_new, y, h_new, c)


def MultivariateGaussianKernel(x, y, h, c=0):
    """
        This function implements the multivariate gaussian kernel.
    
        $K(x,y|h) = \frac{1}{|h|^{1/2}(2\pi)^{d/2}}\exp{-0.5*(x-y)^T h^{-1} (x-y)}$
    
        x,y are D dimensional random vectors.
        x is the sample vector the kernel is built around.
        y is the test vector at which kernel is evaluated.
        y can also be a np.array of random vectors. In this case the result
         will be an array and each element will be the kernel evaluation at the
         corresponding y vector.
        The assumed structure of y is Samples X Dimensions
        The assumed structure of x is 1 X Dimensions
        h is the kernel width matrix. It is a covariance matrix therefore,
         symmetric positive definite.
        c is a null variable.
    """
    deth = np.linalg.det(h)
    invh = np.linalg.inv(h)
    v = y - np.tile(x, [y.shape[0], 1])
    deno = np.sqrt(deth) * (2. * np.pi) ** (np.double(x.shape[1]) / 2.)
    nume = np.sum(np.dot(invh, v.T).T * v, axis=1).reshape([y.shape[0], 1])
    return np.exp(-0.5 * nume) / deno


def MultivariateGaussianKernelMarginal(x, y, h, j, c=0):
    """
        This function implements the marginal integral of a 
        multivariate normal kernel. 
        x, y are D dimensional vectors. 
        x is the sample that forms the center of the kernel. 
        y is the test vector at which kernel is evaluated. 
        y can also be a np.array of random vectors. In this case the result
          will be an array and each element will be the kernel evaluation
          at the corresponding y vector. 
          The assumed structure of y is Samples X Dimensions
          The assumed structure of x is 1 X Dimensions
        h is the kernel width matrix. It is a covariance matrix therefore, 
          it is symmetric positive definite.
        j is the dimension the marginal is computed at. 
          The current implementation only computes a marginal in one
          dimension.
    """
    sigma = h[j, j]
    if y.shape[1] > 1:
        return GaussianKernel(x[:, j].reshape([x.shape[0], 1]),
                              y[:, j].reshape([y.shape[0], 1]),
                              sigma)
    else:
        return GaussianKernel(x[:, j].reshape([x.shape[0], 1]),
                              y, sigma)


## KERNEL DENSITY ESTIMATION ##

def KernelDensityEstimate(kernel, X, y, h, j=-1, c=0):
    """
    This funcion implements the general kernel density estimation

    p(y) = \frac{1}{|X|}\sum_{x \in X} K(x, y | c, h)

    X is a set of samples that are observed already. here the method assumes
      X is a matrix and the first dimension is the sample size.
      
    y is a value at which the density will be evaluated at.
    y can also be a vector and in this case the density will be estimated
      at every element independently.
    
    The assumed structure of y is TestSamples X Dimensions
    The assumed structure of X is TrainingSamples X Dimensions
    
    c is the support of the kernel and depending on the type of the kernel
      it can be a vector or a scalar
      
    h is the width of the kernel and depending on the kernel it can be a
      scalar, a vector or a matrix.
    h can also be a np.array providing different kernels for each sample in X. 
      In this case its first dimension should equal to the number of samples.
    
    kernel is the general kernel function that will be evaluated.
      The inputs of the kernel function should be X, y, h and c.
    
    j is the dimension the marginal is computed at. If j = -1 then the estimate
      will be on the entire space. However, when j > -1 then the estimate
      is only constructed on the jth dimension marginalizing over all other dimensions.
      In this case the kernel should also take j as an argument. 
      The X, y and h on the other hand are the same as the multivariate distribution.
      The current implementation only computes a marginal in one dimension.
    """
    numKernel = X.shape[0]
    p_y = np.zeros([y.shape[0], 1])
    if np.ndim(h) == 0 or (np.ndim(h) > 0 and h.shape[0] != X.shape[0]):
        for n in range(numKernel):
            if j > -1:
                p_y += kernel(X[n, :].reshape([1, X.shape[1]]), y, h, j, c=c) / np.double(numKernel)
            else:
                p_y += kernel(X[n, :].reshape([1, X.shape[1]]), y, h, c=c) / np.double(numKernel)
    else:
        for n in range(numKernel):
            if j > -1:
                p_y += kernel(X[n, :].reshape([1, X.shape[1]]), y, h[n, :], j, c=c) / np.double(numKernel)
            else:
                p_y += kernel(X[n, :].reshape([1, X.shape[1]]), y, h[n, :], c) / np.double(numKernel)
    return p_y


def KernelDensityEstimate_Conditional(kernel, marginalKernel, X, y, h, j, c=0):
    """
    This function implements the general kernel conditional density estimation

    p(y(~j)/y(j)) = \frac{p(y)}{p(y(j))}

    Where p(w,v) is the joint kernel density estimate and p(v) is the marginal 
        kernel density estimate. 

    For kernel, X, y and c see the definition of KernelDensityEstimate. 

    marginalKernel is the marginal distribution corresponding to kernel. 

    j is the vector of indices that are assumed to be given. 
    """
    nume = KernelDensityEstimate(kernel, X, y, h)
    deno = KernelDensityEstimate(marginalKernel, X, y, h, j)
    return nume / deno


## BANDWIDTH ESTIMATION WITH SAMPLES - MAXIMUM LIKELIHOOD APPROACH ##

def LogLikelihood(kernel, X, y, h, c=0):
    """
    This function computes the log likelihood of the kernel density estimator
        constructed using the samples in X on the samples in y.
    It uses the KernelDensityEstimate function given above.
    The kernel is basically selected based on one of the kernels implemented 
        in this piece of code.
    """
    z = KernelDensityEstimate(kernel, X, y, h, c=c)
    logz = np.log(z)

    return sum(logz)


def LogLikelihoodLeaveOneOut(kernel, X, h, c=0):
    """
    This function computes the log likelihood of the kernel density estimator 
        in a leave-one-out fashion. In each computation one point is left out 
        and the likelihood is computed over that point. 
    """
    logz = 0.
    for m in range(X.shape[0]):
        indices = np.ones(X.shape[0], dtype=bool)
        indices[m] = False
        Xn = X[indices, :]
        Y = X[m, :].reshape([1, X.shape[1]])
        logz += LogLikelihood(kernel, Xn, Y, h, c=c)
    return logz


def LogLikelihoodNFoldCV(kernel, X, h, N=5, c=0):
    """
    This function computes the log likelihood of the kernel density estimator 
        in a NFold CV fashion. In each computation numPoints / N points are left out 
        and the likelihood is computed over that point. 
    """
    logz = 0.
    stepSize = np.int(np.double(X.shape[0]) / np.double(N))
    for n in range(N - 1):
        indices = np.ones(X.shape[0], dtype=bool)
        indices[n * stepSize:(n + 1) * stepSize] = False
        Xn = X[indices, :]
        Y = X[~indices, :].reshape([sum(~indices), X.shape[1]])
        logz += LogLikelihood(kernel, Xn, Y, h, c=c)
    indices = np.ones(X.shape[0], dtype=bool)
    indices[(N - 1) * stepSize:] = False

    Xn = X[indices, :]
    Y = X[~indices, :].reshape([sum(~indices), X.shape[1]])
    logz += LogLikelihood(kernel, Xn, Y, h, c=c)
    return logz


def KLDivergence(p_target, p_est, dx=1.):
    """
        This function computes the KL divergence between the 
           the densities p_target and p_est.
        
        p_target and p_est are one-dimensional distributions and
           they are of the same size.
        
        dx is the integration step. In the discrete case user 
           does not need to provide dx.
    """
    if sum(p_est * dx) > 1:
        print('The kernel density estimate sums to larger than 1. Perhaps the spacing is not correct.')
        return -1
    elif sum(p_target * dx) > 1:
        print('The target density sums to larger than 1.')
        return -1

    KL = sum(p_target * np.log(p_target / p_est) * dx)
    return KL


def MahalonobisSPD(M, w):
    """
        This function implements the Mahalonobis distance 
          with the matrix exponential. 
          
        D = w^T exp(-(M + M^T)/2) w
        
        This is useful for Gaussian kernels and gradient updates
          for estimating covariance matrices.
        
        w is a vector of size dim x 1. 
        M is a matrix of size dim x dim.
    """
    A = (M + M.T) / 2.
    expNegA = scipy.linalg.expm(-A)
    return np.dot(w.T, np.dot(expNegA, w))


def DerivativeMahalonobisSPD(M, w, i, j):
    """
        This function implements the derivative
        
        G = \partial(w^T exp(-(M + M^T)/2) w) / \partial M_{ij}.
        
        This is crucial for the gradient updates when optimizing for a 
        covariance matrix in a Gaussian kernel. 
        
        Here w is a vector of size dim x 1 or a matrix of size dim x samples
        M is a matrix of size dim x dim. 
        M+M^T is always symmetric. 
        exp(-(M+M^T)/2) is always symmetric positive definite.
    """
    A = (M + M.T) / 2.
    expNegA = scipy.linalg.expm(-A)
    W = -1. * np.dot(np.dot(expNegA, w), np.dot(w.T, expNegA))
    # print DerivativeSPD(M, i, j)
    return np.trace(np.dot(W, DerivativeSPD(M, i, j)))


def DerivativeSPD(M, i, j):
    """
        This function implements the derivative
        
        G = \partial exp((M + M^T)/2)\partial M_{ij}
        
        This is derivative is useful for computing gradient updates
          in covariance matrix estimation.
          
        M is a matrix of size dim x dim. 
        The output will also be a matrix.
    """
    A = (M + M.T) / 2.
    # print 'A:{0}'.format(A)

    s, U = np.linalg.eig(A)  # will return the eigenvalue decomposition even if 0's exist.
    # print 's:{0}'.format(s)
    # print 'U:{0}'.format(U)

    xi = U[i, :].reshape([1, M.shape[1]])
    xj = U[j, :].reshape([1, M.shape[1]])
    f = np.dot(xi.T, xj) / 2. + np.dot(xj.T, xi) / 2.
    # print 'f:{0}'.format(f)
    Vu = np.zeros(M.shape)
    for m in range(Vu.shape[0]):
        for n in range(Vu.shape[1]):
            if (m == n) or (s[m] == s[n]):
                Vu[m, n] = f[m, n] * np.exp(s[m])
            else:
                Vu[m, n] = f[m, n] * (np.exp(s[m]) - np.exp(s[n])) / (s[m] - s[n])
    # print 'Vu:{0}'.format(Vu)
    G = np.dot(U, np.dot(Vu, U.T))
    return G


def ConvertToCovariance(M, mtype='multivariate'):
    """
        If mtype (kernel type) is multivariate then
        This function implements the conversion from M to H: 
        
        H = exp((M + M^T)/2)
        
        The input M can be an array of matrices as well as a 
          single matrix. The output will be accordingly an 
          array or a single matrix.
        The exponential is a matrix exponential

        If mtype is univariate or product then 
        This function implements the conversion from M to H: 

        H = exp(M)
    """
    if mtype == 'multivariate':
        if np.ndim(M) == 3:  # means that there is an array of cov matrices
            H = np.zeros(M.shape)
            for k in range(M.shape[0]):
                H[k, :] = scipy.linalg.expm((M[k, :] + M[k, :].T) / 2.)
        else:  # means that there is only one matrix
            H = scipy.linalg.expm((M + M.T) / 2.)
        return H
    else:
        if np.ndim(M) == 2:  # means that there is an array of width vectors
            H = np.zeros(M.shape)
            for k in range(M.shape[0]):
                H[k, :] = np.exp(M[k, :])
        else:
            H = np.exp(M)
        return H


def LogLikelihoodGradient_MultivariateGaussianKernel(X, y, M, k=-1):
    """
        This function implements the gradient of the loglikelihood
            of the estimated density with respect to the bandwidth h. 
            The kernel used in this implementation is the multivariate 
            Gaussian kernel. The loglikelihood is computed using the points in y.
        
        Please check the article for further details on this term.

        See KernelDensityEstimate for the definitions of the variables: 
            X and y.
            
        M = log(h), h is from KernelDensityEstimate function. 
            This uses the differential structure on the Lie group
            of symmetric positive definite matrices, i.e. covariance 
            matrices. M lives in the Lie algebra, which is the
            space of symmetric matrices, a vector space. M is supposed 
            to be symmetric all the time. This enforces, the gradient of M
            is to be symmetric as well if a basic gradient descent is 
            going to be applied.
        
        M can be a single matrix or an np.array of matrices, which
            indicates there is a different kernel for each sample. If
            M is a single matrix then the gradient will be taken with
            respect to that point. If M is an np.array of matrices then
            k indicates the dimension the gradient will be taken with
            respect to.
            When M is a single matrix its dimensions should be numDims x numDims. 
            When M is an np.array of matrices then its dimension is 
                numSamples x numDims x numDims.
    """
    if np.ndim(M) == 3:  # means there is an array of covariance matrices
        numKernel = np.double(X.shape[0])
        Mk = M[k, :]
        G = np.zeros(Mk.shape)
        H = ConvertToCovariance(M)
        p_H = KernelDensityEstimate(MultivariateGaussianKernel, X, y, H)
        Xk = X[k, :].reshape([1, X.shape[1]])
        K_Hk = MultivariateGaussianKernel(Xk, y, H[k, :])
        alpha_sq = K_Hk / p_H / numKernel
        alpha = np.sqrt(alpha_sq)
        W = np.tile(alpha.T, [X.shape[1], 1]) * (y.T - np.tile(Xk.T, [1, y.shape[0]]))
        firstTerm = np.sum(alpha_sq)
        for m in range(G.shape[0]):
            for n in range(G.shape[1]):
                if m == n:
                    G[m, n] = -1. / 2. * (firstTerm + DerivativeMahalonobisSPD(Mk, W, m, n))
                else:
                    G[m, n] = -1. / 2. * (DerivativeMahalonobisSPD(Mk, W, m, n))
        return G

    else:  # means there is only one covariance matrix
        numKernel = np.double(X.shape[0])
        G = np.zeros(M.shape)
        H = ConvertToCovariance(M)
        for k in range(np.int(numKernel)):
            p_H = KernelDensityEstimate(MultivariateGaussianKernel, X, y, H)
            Xk = X[k, :].reshape([1, X.shape[1]])
            K_Hk = MultivariateGaussianKernel(Xk, y, H)
            alpha_sq = K_Hk / p_H / numKernel
            alpha = np.sqrt(alpha_sq)
            W = np.tile(alpha.T, [X.shape[1], 1]) * (y.T - np.tile(Xk.T, [1, y.shape[0]]))
            # print W
            firstTerm = np.sum(alpha_sq)
            for m in range(G.shape[0]):
                for n in range(G.shape[1]):
                    # print m, n
                    if m == n:
                        G[m, n] += -1. / 2. * (firstTerm + DerivativeMahalonobisSPD(M, W, m, n))
                    else:
                        G[m, n] += -1. / 2. * (DerivativeMahalonobisSPD(M, W, m, n))
        return G


def LogLikelihoodGradient_GaussianProductKernel(X, y, M, k=-1):
    """
    This function implements the loglikelihood gradient for kernel 
    density estimation using gaussian product kernel. 

    For definitions of X and y please see KernelDensityEstimate function. 

    M = log(H) pointwise. 

    """
    if np.ndim(M) == 2:  # means there is an array of width vectors.
        numKernel = np.double(X.shape[0])
        Mk = M[k, :]
        H = ConvertToCovariance(M, mtype='product')
        p_H = KernelDensityEstimate(GaussianProductKernel, X, y, H)
        G = np.zeros(Mk.shape)
        Xk = X[k, :].reshape([1, X.shape[1]])
        K_Hk = GaussianProductKernel(Xk, y, H[k, :])
        alpha = K_Hk / p_H / numKernel
        for m in range(G.shape[0]):
            yv = y[:, m].reshape([y.shape[0], 1])
            xv = np.tile(Xk[:, m], [y.shape[0], 1])
            G[m] += -1. / 2. * sum((1. - (xv - yv) ** 2 / H[k, m]) * alpha)
        return G
    else:
        numKernel = np.double(X.shape[0])
        H = ConvertToCovariance(M, mtype='product')
        p_H = KernelDensityEstimate(GaussianProductKernel, X, y, H)
        G = np.zeros(M.shape)
        for k in range(np.int(numKernel)):
            Xk = X[k, :].reshape([1, X.shape[1]])
            K_Hk = GaussianProductKernel(Xk, y, H)
            alpha = K_Hk / p_H / numKernel
            for m in range(G.shape[0]):
                yv = y[:, m].reshape([y.shape[0], 1])
                xv = np.tile(Xk[:, m], [y.shape[0], 1])
                G[m] += -1. / 2. * sum((1. - (xv - yv) ** 2 / H[m]) * alpha)
        return G


def LogLikelihoodLeaveOneOutGradient_GPK(X, M, k=-1):
    """
    This function implements the gradient of the LogLikelihoodLeaveOneOut Cost function
        with the GaussianProductKernel kernel type. 
    """
    G = np.zeros(M.shape)

    for m in range(X.shape[0]):
        indices = np.ones(X.shape[0], dtype=bool)
        indices[m] = False
        Xn = X[indices, :]
        Y = X[m, :].reshape([1, X.shape[1]])
        G += LogLikelihoodGradient_GaussianProductKernel(Xn, Y, M, k=k)
    return G


def LogLikelihoodNFoldCVGradient_GPK(X, M, N=5, k=-1):
    G = np.zeros(M.shape)

    stepSize = np.int(np.double(X.shape[0]) / np.double(N))
    for n in range(N - 1):
        indices = np.ones(X.shape[0], dtype=bool)
        indices[n * stepSize:(n + 1) * stepSize] = False
        Xn = X[indices, :]
        Y = X[~indices, :].reshape([sum(~indices), X.shape[1]])
        G += LogLikelihoodGradient_GaussianProductKernel(Xn, Y, M, k=k)
    indices = np.ones(X.shape[0], dtype=bool)
    indices[(N - 1) * stepSize:] = False

    Xn = X[indices, :]
    Y = X[~indices, :].reshape([sum(~indices), X.shape[1]])
    G += LogLikelihoodGradient_GaussianProductKernel(Xn, Y, M, k=k)
    return G


def EstimateBandwidth(X, Y, kernel, Cost, CostGradient, stepSize=0.01, ktype='single', mtype='multivariate'):
    """
        This function implements the optimization routine to 
          estimate the kernel bandwidth. It uses gradient ascent 
          on the cost function. 
        
        X are the samples given as a matrix of samples x dimensions. 
        Y are the test samples given as a matrix of samples x dimensions.
        Y can equal to X.
        
        kernel is the kernel used in the estimation.
        cost is the cost function, e.g. log likelihood. 
        costGradient is the gradient of the cost function. 
        
        stepSize is the multiplicative factor for the ascent step. 
        
        ktype='single' or 'multiple' indicates whether a single bandwidth or 
          multiple bandwidths will be estimated. If single then the estimated
          bandwidth is shared across all the kernels situated around each sample.
          If multiple then a different bandwidth will be estimated for each kernel. 
          IMPORTANT NOTE: estimating multiple kernels require a lot harder and prone 
          to undesired results. Especially when the points are sparse. Then some points 
          might be isolated and kernels on these points will tend towards Dirac's delta. 
    """
    # gradient ascent routine

    numKernels = X.shape[0]
    numDim = X.shape[1]
    if mtype == 'multivariate':
        if ktype == 'single':
            M = np.random.normal(loc=0, scale=1, size=[numDim, numDim])
            M_next = np.zeros([numDim, numDim])
        else:
            M = np.zeros([numKernels, numDim, numDim])
            M_next = np.zeros([numKernels, numDim, numDim])
            for k in range(numKernels):
                M[k, :] = np.random.normal(loc=0, scale=1, size=[numDim, numDim])
    else:
        if ktype == 'single':
            M = np.random.normal(loc=0, scale=1, size=[numDim])
            M_next = np.zeros([numDim])
        else:
            M = np.zeros([numKernels, numDim])
            M_next = np.zeros([numKernels, numDim])
            for k in range(numKernels):
                M[k, :] = np.random.normal(loc=0, scale=1, size=[numDim])

    numMaxIter = 500
    cost_value = np.zeros(numMaxIter + 1)
    H = ConvertToCovariance(M, mtype=mtype)
    cost_value[0] = Cost(kernel, X, Y, H)

    for it in range(numMaxIter):
        Grad = CostGradient(X, Y, M)
        if ktype == 'single':
            M_next = M + stepSize * Grad
            M = M_next.copy()
        else:
            for k in range(numKernels):
                Grad = CostGradient(X, Y, M, k=k)
                M_next[k, :] = M[k, :] + stepSize * Grad
            for k in range(numKernels):
                M[k, :] = M_next[k, :]
        H = ConvertToCovariance(M, mtype=mtype)
        cost_value[it + 1] = Cost(kernel, X, Y, H)
        if it % 20 == 0:
            print('Iteration:{0} - Cost:{1} - h:{2}'.format(it, cost_value[it + 1], H))
        cost_value_gain = (cost_value[it + 1] - cost_value[it]) / np.abs(cost_value[it])
        if cost_value_gain > 0. and cost_value_gain < 1.e-10:
            break
    print('Final Cost: {0} at iteration {1}'.format(cost_value[it + 1], it))
    return H


def EstimateBandwidth_CV(X, kernel, Cost, CostGradient, N=5, stepSize=0.01, ktype='single', mtype='multivariate'):
    """
        This function implements the optimization routine to 
          estimate the kernel bandwidth. It uses gradient ascent 
          on the cost function. 

        The main objective functions are Cross-validation type. 
        
        X are the samples given as a matrix of samples x dimensions. 
                
        
        kernel is the kernel used in the estimation.
        cost is the cost function, e.g. log likelihood. 
        costGradient is the gradient of the cost function. 
        
        stepSize is the multiplicative factor for the ascent step. 
        
        ktype='single' or 'multiple' indicates whether a single bandwidth or 
          multiple bandwidths will be estimated. If single then the estimated
          bandwidth is shared across all the kernels situated around each sample.
          If multiple then a different bandwidth will be estimated for each kernel. 
          IMPORTANT NOTE: estimating multiple kernels require a lot harder and prone 
          to undesired results. Especially when the points are sparse. Then some points 
          might be isolated and kernels on these points will tend towards Dirac's delta. 
    """
    # gradient ascent routine

    numKernels = X.shape[0]
    numDim = X.shape[1]
    if mtype == 'multivariate':
        if ktype == 'single':
            M = np.random.normal(loc=0, scale=1, size=[numDim, numDim])
            M_next = np.zeros([numDim, numDim])
        else:
            M = np.zeros([numKernels, numDim, numDim])
            M_next = np.zeros([numKernels, numDim, numDim])
            for k in range(numKernels):
                M[k, :] = np.random.normal(loc=0, scale=1, size=[numDim, numDim])
    else:
        if ktype == 'single':
            # M = np.random.normal(loc=0,scale=1,size=[numDim])
            M = np.log(np.std(X, axis=0))
            M_next = np.zeros([numDim])
        else:
            M = np.zeros([numKernels, numDim])
            M_next = np.zeros([numKernels, numDim])
            for k in range(numKernels):
                # M[k,:] = np.random.normal(loc=0,scale=1,size=[numDim])
                M[k, :] = np.log(np.std(X, axis=0))

    numMaxIter = 500
    cost_value = np.zeros(numMaxIter + 1)
    H = ConvertToCovariance(M, mtype=mtype)
    cost_value[0] = Cost(kernel, X, H, N=N)

    for it in range(numMaxIter):
        Grad = CostGradient(X, M, N=N)
        if ktype == 'single':
            M_next = M + stepSize * Grad
            M = M_next.copy()
        else:
            for k in range(numKernels):
                Grad = CostGradient(X, M, k=k, N=N)
                M_next[k, :] = M[k, :] + stepSize * Grad
            for k in range(numKernels):
                M[k, :] = M_next[k, :]
        H = ConvertToCovariance(M, mtype=mtype)
        cost_value[it + 1] = Cost(kernel, X, H, N=N)
        if it % 100 == 0 and it > 0:
            print('Iteration:{0} - Cost:{1} - h:{2}'.format(it, cost_value[it + 1], H))
        cost_value_gain = (cost_value[it + 1] - cost_value[it]) / np.abs(cost_value[it])
        if cost_value_gain > 0. and cost_value_gain < 1.e-6:
            break
    print('Final Cost: {0} at iteration {1}'.format(cost_value[it + 1], it))
    return H
