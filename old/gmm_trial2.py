from typing import Tuple

import numpy as np
import scipy.stats
import scipy.linalg
import scipy.io
from sklearn import datasets


class GMM:
    def __init__(self, K, max_iter=300, tol=1e-4):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol

        self.means = None
        self.covs = None
        self.weights = None
        self.tau_ik = None
        self.ll = []

    def init_means(self, data):
        self.means = np.random.normal(size=(self.K, data.shape[1]))

    def init_covs(self, data):
        self.covs = np.repeat(np.identity(data.shape[1])[np.newaxis, :, :], self.K, axis=0)

    def init_weights(self):
        self.weights = np.repeat(1 / self.K,  self.K)

    def init_tau(self, data):
        self.tau_ik = np.full((data.shape[0], self.K), 1 / self.K)

    def update_tau(self, k, data):
        self.tau_ik[:, k] = self.tau_ik[:, k] * scipy.stats.multivariate_normal.pdf(
            data,
            mean=self.means[k],
            cov=self.covs[k]
        )

    def get_loglikelihood(self):
        self.ll.append(np.sum(np.log(np.sum(self.tau_ik, axis=1))))

    def normalize_tau(self):
        self.tau_ik = self.tau_ik / self.tau_ik.sum(axis=1, keepdims=1)

    def update_means(self, k, data):
        if self.means.shape[1] != data.shape[1]:
            self.means = np.zeros((k, data.shape[1]))
        self.means[k, :] = np.dot(self.tau_ik[:, k], data) / self.tau_ik[:, k].sum(0)

    def update_covs(self, k, data):
        if self.covs.shape[2] != data.shape[1]:
            self.covs = np.zeros((self.k, data.shape[1], data.shape[1]))
        res = []
        for i in range(data.shape[0]):
            res.append(self.tau_ik[i, k] * (data[i, :] - self.means[k]) * (data[i, :] - self.means[k]).reshape(-1, 1))
        self.covs[k] = np.array(res).sum(0)

    def update_weights(self, k, data):
        self.weights[k] = self.tau_ik[:, k].sum(0) / data.shape[0]

    @staticmethod
    def get_r_eigvals_eigvecs(cov, r):
        eig_vals, eig_vecs = scipy.linalg.eig(cov)
        sort_idx = np.argsort(eig_vals)
        sorted_vecs = eig_vecs[:, sort_idx].real
        sorted_vals = eig_vals[sort_idx].real
        return sorted_vals[:r], sorted_vecs[:, :r]

    @staticmethod
    def get_cov_approx(eigvals, eigvecs):
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    @staticmethod
    def transform_x(data, eigvecs):
        return data @ eigvecs

    def transform_mean(self, k, eigvecs):
        # return self.means[k] @ eigvecs
        return eigvecs.T @ self.means[k]

    def low_rank_update_tau(self, k, data, eigvals, eigvecs):
        x_transform = self.transform_x(data, eigvecs)
        mean_transform = self.transform_mean(k, eigvecs)
        # res = []
        # for i in range(data.shape[0]):
        #     res.append((((x_transform[i, :] - mean_transform[k])**2) / eigvals[k]))k
        # m_k = np.array(res).sum()
        m_k = (((x_transform - mean_transform) ** 2) / eigvals).sum()
        d_k = np.product(eigvals**(-1/2))
        self.tau_ik[:, k] = self.weights[k] * d_k * np.exp((-1/2) * m_k)

    def fit(self, data):
        self.init_means(data)
        self.init_covs(data)
        self.init_weights()
        self.init_tau(data)

        curr_iter = 0
        converged = False
        while not converged and curr_iter < self.max_iter:
            for k in range(self.K):
                self.update_tau(k, data)
            self.normalize_tau()
            for k in range(self.K):
                self.update_means(k, data)
                self.update_covs(k, data)
                self.update_weights(k, data)
            self.get_loglikelihood()
            if curr_iter > 1:
                converged = abs(self.ll[curr_iter] - self.ll[curr_iter - 1]) < self.tol
            curr_iter += 1

  
def load_data_labels(data_file: str, label_file: str) -> Tuple[np.ndarray, np.ndarray]:
    data_contents = scipy.io.loadmat(data_file)
    label_contents = scipy.io.loadmat(label_file)
    return data_contents['data'].T, label_contents['trueLabel'].T


def main():
    iris = datasets.load_iris()
    data = iris.data
    labels = iris.target
    gmm = GMM(3)
    gmm.fit(data)


if __name__ == '__main__':
    main()
