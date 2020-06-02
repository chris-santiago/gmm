import numpy as np
from sklearn import datasets
import scipy.stats


def get_r_eigvals_eigvecs(cov, r):
    eig_vals, eig_vecs = scipy.linalg.eig(cov)
    sort_idx = np.argsort(eig_vals)[::-1]  # descending order
    sorted_vecs = eig_vecs[:, sort_idx].real
    sorted_vals = eig_vals[sort_idx].real
    return sorted_vals[:r], sorted_vecs[:, :r]


iris = datasets.load_iris()
data = iris.data
labels = iris.target

K = 3
R = 3
# means = np.random.normal(size=(K, data.shape[1]))
means = np.array([  # cheating means for testing purposes
    [5.006, 3.428, 1.462, 0.246],
    [5.936, 2.77, 4.26, 1.326],
    [6.588, 2.974, 5.552, 2.026]
])
covs = np.repeat(np.identity(data.shape[1])[np.newaxis, :, :], K, axis=0)
weights = np.repeat(1 / K, K)
resp = np.full((data.shape[0], K), 1 / K)
log_likelihood = np.sum(np.log(np.sum(resp, axis=1)))
likelihoods = [log_likelihood]

curr_iter = 0
converged = False
while not converged and curr_iter < 100:

    for k in range(K):
        eigvals, eigvecs = get_r_eigvals_eigvecs(covs[k], R)
        sigma_approx = eigvecs @ np.diag(eigvals) @ eigvecs.T
        data_approx = data @ eigvecs
        mu_approx = means[k] @ eigvecs
        resp[:, k] = weights[k] * (1 / np.sqrt(np.prod(eigvals))) * np.exp((-1/2) * np.sum(((data_approx - mu_approx) ** 2) / eigvals, axis=1))

    log_likelihood = np.sum(np.log(np.sum(resp, axis=1)))
    likelihoods.append(log_likelihood)

    # normalize
    resp = resp / resp.sum(axis=1, keepdims=1)

    for k in range(K):
        means[k, :] = np.dot(resp[:, k], data) / resp[:, k].sum()
        res = []
        for i in range(data.shape[0]):
            res.append(resp[i, k] * (data[i, :] - means[k]) * (data[i, :] - means[k]).reshape(-1, 1))
        covs[k] = np.array(res).sum(0) / resp[:, k].sum()
        weights[k] = resp[:, k].sum(0) / data.shape[0]

    if curr_iter > 1:
        converged = abs(likelihoods[curr_iter] - likelihoods[curr_iter-1]) < 1e-4
    curr_iter += 1
