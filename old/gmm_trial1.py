import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List, Optional
from sklearn import datasets
import scipy.stats
from sklearn.preprocessing import StandardScaler


def load_data_labels(data_file: str, label_file: str) -> Tuple[np.ndarray, np.ndarray]:
    data_contents = scipy.io.loadmat(data_file)
    label_contents = scipy.io.loadmat(label_file)
    return data_contents['data'].T, label_contents['trueLabel'].T


class GMM:
    def __init__(self, data: np.ndarray, k: int = 2):
        self.data = data
        self.k = k
        self._n_feat = data.shape[1]

    def get_initial_means(self) -> np.ndarray:
        means = []
        for i in range(self.k):
            means.append(np.random.normal(size=self._n_feat))
        return np.stack(means)

    def get_initial_covariances(self) -> np.ndarray:
        covs = []
        for i in range(self.k):
            covs.append(np.identity(self._n_feat))
        return np.stack(covs)

    def get_initial_densities(self, means: np.ndarray, covs: np.ndarray) -> np.ndarray:
        densities = []
        for i in range(self.k):
            kde = scipy.stats.multivariate_normal.pdf(self.data,
                                                      mean=means[i],
                                                      cov=covs[i])
            densities.append(kde)
        return np.stack(densities)

    @staticmethod
    def get_eigvals_eigvecs(matrix: np.ndarray,
                            ascending: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Returns sorted eigenvalues and eigenvectors"""
        eig_vals, eig_vecs = scipy.linalg.eig(matrix)
        if ascending:
            sort_idx = np.argsort(eig_vals)
        else:
            sort_idx = np.argsort(eig_vals)[::-1]
        sorted_vecs = eig_vecs[:, sort_idx].real
        sorted_vals = eig_vals[sort_idx].real
        return sorted_vals, sorted_vecs

    @staticmethod
    def get_low_rank_sigma(eigvals: np.ndarray, eigvecs: np.ndarray, rank: int) -> np.ndarray:
        """
        Compute the low-rank approximation of the covariance matrix.

        When the covariance matrix is rank-deficient (i.e. eigenvalues ~ 0) it cannot be
        inverted.
        """
        return eigvecs[:, :rank] @ np.diag(eigvals[:rank]) @ eigvecs[:, :rank].T

    def get_density_approx(self):
        pass

    def fit(self, rank: int, max_iter: int = 300):
        pass
        means = self.get_initial_means()
        covs = self.get_initial_covariances()
        densities = self.get_initial_densities(means, covs)
        mix = [0.5, 0.5]

        for k in range(self.k):
            i = 0
            while not self.is_converged() and i < max_iter:
                tau = self.evaluate_responsibilities()
                self.update_mu()
                self.update_sigma()
                self.update_mix()
                self.evaluate_log_likelihood()
                self.is_converged()
                i += 1


def main():
    data, labels = load_data_labels('homework2/data/data.mat', 'homework2/data/label.mat')

    im1 = data[0].reshape(28, 28, order='F')
    im2 = data[-1].reshape(28, 28, order='F')

    plt.imshow(im1, cmap='gray')
    plt.imshow(im2, cmap='gray')


def test():
    iris = datasets.load_iris()
    data = iris.data
    labels = iris.target
    k = 3

    gmm = GMM(data, k)
    means = gmm.get_initial_means()
    covs = gmm.get_initial_covariances()
    densities = gmm.get_initial_densities(means, covs)
    weights = np.repeat(1 / 3, 3)
    responsibilties = np.zeros((data.shape[0], k))
    j = 0
    while j < 30:
        priors = responsibilties
        for i in range(k):
            # calc responsibilities
            responsibilties[:, i] = (weights[i] * densities[i]) / (
                        weights[:, np.newaxis] * densities).sum(0)
            # calc new mean
            means[i, :] = (responsibilties[:, i].reshape(-1, 1) * data).sum(0) / responsibilties[:,
                                                                                 i].sum()
            # calc new cov
            res = []
            for n in range(data.shape[0]):
                res.append(
                    responsibilties[n, i] * (data[n, :] - means[i]) * (
                                data[n, :] - means[i]).reshape(
                        -1, 1))
            covs[i] = np.array(res).sum(0)
            # calc new mix
            weights[i] = (responsibilties[:, i].sum()) / data.shape[0]
        densities = gmm.get_initial_densities(means, covs)
        print(f'j={j}')
        print(responsibilties)
        j += 1


def test_2():
    iris = datasets.load_iris()
    data = iris.data
    labels = iris.target
    k = 3

    def gmm_func(data, K, max_iter=600):
        # initialize parameters
        tau_ik = np.repeat(1 / 3, data.shape[0] * K).reshape(-1, K)
        weights = np.repeat(1 / 3, 3)
        means = np.random.normal(size=(K, data.shape[1]))
        covs = np.repeat(np.identity(data.shape[1])[np.newaxis, :, :], 3, axis=0)

        prior_ll, curr_iter = 0, 0
        not_converged = True
        while not_converged and curr_iter < max_iter:
            # update tau_ik
            for k in range(K):
                tau_ik[:, k] = tau_ik[:, k] * scipy.stats.multivariate_normal.pdf(data,
                                                                                  mean=means[k],
                                                                                  cov=covs[k])

            # get log likelihood
            curr_ll = np.sum(np.log(np.sum(tau_ik, axis=1)))

            # normalize tau_ik
            tau_ik = tau_ik / tau_ik.sum(axis=1, keepdims=1)

            # check convergence
            not_converged = abs(curr_ll - prior_ll) > 1e-7

            # update params
            for k in range(K):
                # calc new means
                means[k, :] = np.dot(tau_ik[:, k], data) / tau_ik[:, k].sum(0)

                # calc new covs
                res = []
                for i in range(data.shape[0]):
                    res.append(
                        tau_ik[i, k] * (data[i, :] - means[k]) * (data[i, :] - means[k]).reshape(
                            -1, 1))
                covs[k] = np.array(res).sum(0)

                # calc new weights
                weights[k] = tau_ik[:, k].sum(0) / data.shape[0]

            # reset prior_ll value, increment iterations
            prior_ll = curr_ll
            curr_iter += 1
        return tau_ik

    print(gmm_func(data, K=k))


if __name__ == '__main__':
    data, labels = load_data_labels('homework2/data/data.mat', 'homework2/data/label.mat')

    im1 = data[0].reshape(28, 28, order='F')
    im2 = data[-1].reshape(28, 28, order='F')
