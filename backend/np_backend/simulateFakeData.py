import numpy as np
from scipy.stats import norm, ncx2
from np_backend.numpy_utils import entropy, numpy_softmax

def simulateLogisticRegressionData(
    shape=(1000, 64),
    nClasses=10,
    noisyFeatureOccurrence=0.1,
    noisyFeatureSampling=1.0,
    tol=1e-50,
):
    N, d = shape
    K = nClasses

    noisyFeatures = np.random.binomial(1, p=noisyFeatureOccurrence, size=d)
    shape_theta = np.where(noisyFeatures, 0.2, 0.01)
    theta_exp = np.random.gamma(
        shape=shape_theta[:, np.newaxis] * np.ones((d, K))
    )  # Shape dxK
    theta_exp = theta_exp / theta_exp.max(axis=1, keepdims=True)
    probas = (theta_exp + tol / d) / (theta_exp.sum(axis=1, keepdims=True) + tol)

    print(probas.sum(axis=1))
    assert np.allclose(probas.sum(axis=1), 1.0)
    theta = np.log(probas)

    entropies = entropy(probas)
    order_by_entropy = np.argsort(entropies)
    theta = theta[order_by_entropy, :]
    noisyFeatures = noisyFeatures[order_by_entropy]
    probas = probas[order_by_entropy, :]

    X, y = simulateDiscreteXYfromProbas(
        shape, probas, ratioXsampling=noisyFeatures * noisyFeatureSampling + np.ones(d)
    )
    return X, y, theta, noisyFeatures


# Purposedly confuses classes 1 and 2 but not class 3
# X is sampled discretely
def simulateDiscreteConfusedData(
    shape=(1000, 64), 
    nClasses=3, 
    noisyFeatureOccurrence=0.5, 
    noisyFeatureSampling=1.0,
    tol=1e-10
):
    N, d = shape
    K = nClasses

    noisyFeatures = np.random.binomial(1, p=noisyFeatureOccurrence, size=d)
    probas = tol * np.ones((d, K))
    probas[(noisyFeatures == 1.0), :2] = 0.5
    probas[np.where(noisyFeatures == 0.0), :] = 0.01 / (K - 1)
    probas[
        np.where(noisyFeatures == 0.0)[0],
        np.random.choice(K, int(np.sum(noisyFeatures == 0.0))),
    ] = 0.99
    probas = probas / probas.sum(axis=1, keepdims=True)
    theta = np.log(probas)
    probas, noisyFeatures, theta = orderProbasByEntropy(probas, noisyFeatures, theta)
    
    X, y = simulateDiscreteXYfromProbas(
        shape, probas, ratioXsampling=noisyFeatures * noisyFeatureSampling + np.ones(d)
    )
    return X, y, theta, noisyFeatures

# Simulate a (sort of) probit model 
# Each Xi is N(0,Id)
# Each Yi is Categorical(\Phi(Xi^Theta)_k)
# Theta[:,k] represents the vector for class k
# Theta[:,0] and Theta[:,1] are voluntarly close
def simulateContinuousConfusedData_Deprecated(
    shape=(1000, 64), 
    nClasses=3,
    eps=1e-2,
):
    N,d = shape
    theta = np.zeros((d,nClasses))
    theta[:2,0] = np.array([1,eps])/np.sqrt(1+eps**2)
    theta[:2,1] = np.array([1,-eps])/np.sqrt(1+eps**2)
    theta[:,2:] = np.random.randn(d, nClasses-2)
    theta[:,2:] = theta[:,2:] / np.sqrt((theta[:,2:]**2).sum(axis=0, keepdims=True))
    
    X = np.random.randn(N,d)
    py_x = np.log(norm.cdf(X @ theta))
    py_x = py_x - py_x.max(axis=1, keepdims=True)
    py_x = np.exp(py_x) 
    py_x = py_x / py_x.sum(axis=1, keepdims=True)
    
    u = np.random.rand(N, 1)
    y = (u > py_x.cumsum(axis=1)).sum(axis=1)
    
    return X, y, theta
    
# Simulate a (sort of) probit model 
# Each Xi is N(0,Id)
# Each Yi is Categorical(\Phi(Xi^Theta)_k)
# Theta[:,k] represents the vector for class k
# Theta[:,0] and Theta[:,1] are voluntarly close
def simulateContinuousConfusedData(
    shape=(1000, 64), 
    eps_confusion=1.0,
    n_classes=3,
):
    N,d = shape
    
    theta = np.zeros((d,n_classes))
    theta[0,0] = 1.0
    theta[0,1] = 1.0
    theta[d-1,0] = eps_confusion
    theta[d-1,1] = - eps_confusion
    theta[1,2:] = 1.0
    theta = np.sqrt(d) * theta
    
    X = np.random.randn(N,d)
    py_x = numpy_softmax(X @ theta)
    u = np.random.rand(N, 1)
    y = (u > py_x.cumsum(axis=1)).sum(axis=1)
    
    return X, y, theta

# Orders all matrices by increasing entropy 
# probas is a dxk matrix where each line is
# a probability distribution
def orderProbasByEntropy(probas, noisyFeatures, theta):
    entropies = entropy(probas)
    order_by_entropy = np.argsort(entropies)
    return probas[order_by_entropy, :], noisyFeatures[order_by_entropy], theta[order_by_entropy, :]
    
# Returns X discrete as vectors from the standard basis
# Y is categorical with P(y=k|x=ej) = probas[j,k] 
def simulateDiscreteXYfromProbas(shape, probas, ratioXsampling):
    N, d = shape

    X = np.zeros(shape)
    indices = np.random.choice(d, size=N, p=ratioXsampling / ratioXsampling.sum())
    X[np.arange(N), indices] = 1.0

    py_x = probas[indices, :]
    u = np.random.rand(N, 1)
    y = (u > py_x.cumsum(axis=1)).sum(axis=1)

    return X, y


def simulateGaussianMixture(prior_probs, mus, sigma=1.0, covs=None, n_samples=1):
    if covs is not None:
        raise NotImplemented()
    n_classes = len(prior_probs)
    assert(mus.shape[0] == n_classes)
    n_features = mus.shape[1]
    
    labels = np.random.choice(n_classes, size=n_samples, p=prior_probs)
    
    if np.array(sigma).ndim==0:
        sigma = np.repeat(sigma, n_classes)[:,np.newaxis]
    elif np.array(sigma).ndim==1:
        sigma = np.array(sigma)[:,np.newaxis]
        
    features = mus[labels,:] + sigma[labels] * np.random.randn(n_samples, n_features)
    
    return features, labels
    
def simulate_logistic_regression_with_hidden_variables(n_samples, n_classes, n_features, edges, taus, magnitude_coefs):
    n_hidden_variables = edges.shape[0]
    
    features = np.random.randn(n_samples, n_features)
    
    coefs    = np.random.randn(n_classes, n_features)
    coefs    = magnitude_coefs * coefs / np.sqrt(np.sum(coefs**2, axis=1, keepdims=True))
    
    hidden_variables = 2 * np.random.binomial(1, 0.5, size=(n_samples, n_hidden_variables)) - 1
    
    intercepts = 1. #5 * np.abs(np.random.randn(n_classes))
    
    py_xh = features @ coefs.T + intercepts
    
    for e in range(n_hidden_variables):
        py_xh[:,edges[e,0]] += taus[e] * hidden_variables[:,e]
        py_xh[:,edges[e,1]] += taus[e] * hidden_variables[:,e]
        
    py_xh  = np.exp(-py_xh) / (np.exp(py_xh) + np.exp(-py_xh))
    
    labels = 1.0 * (np.random.rand(n_samples, n_classes) < py_xh)
    return features, labels, hidden_variables, coefs, intercepts
    
    