from gmm.base_gmm import BaseGMM, load_data_labels
import numpy as np
import scipy.linalg
import scipy.io
from sklearn import datasets


class LowRankGMM(BaseGMM):
    def __init__(self, k, r, tol=1e-4, max_iter=100):
        super().__init__(k, tol, max_iter)
        self.K = k
        self.R = r
        self.tol = tol
        self.max_iter = max_iter

        self.means = None
        self.covs = None
        self.weights = None
        self.resp = None
        self.likelihoods = []

    @staticmethod
    def get_r_eigvals_eigvecs(cov, r):
        eig_vals, eig_vecs = scipy.linalg.eig(cov)
        sort_idx = np.argsort(eig_vals)[::-1]  # descending order
        sorted_vecs = eig_vecs[:, sort_idx].real
        sorted_vals = eig_vals[sort_idx].real
        return sorted_vals[:r], sorted_vecs[:, :r]

    def e_step(self, data):
        for k in range(self.K):
            eigvals, eigvecs = self.get_r_eigvals_eigvecs(self.covs[k], self.R)
            sigma_approx = eigvecs @ np.diag(eigvals) @ eigvecs.T
            data_approx = data @ eigvecs
            mu_approx = self.means[k] @ eigvecs
            self.resp[:, k] = self.weights[k] * (1 / np.sqrt(np.prod(eigvals))) * np.exp(
                (-1 / 2) * np.sum(((data_approx - mu_approx) ** 2) / eigvals, axis=1))
        self.normalize_resp()


def main_iris():
    iris = datasets.load_iris()
    data = iris.data
    labels = iris.target

    gmm = LowRankGMM(k=3, r=3)
    gmm.fit(data)
    print(gmm.plot_likelihoods())


def main():
    data, labels = load_data_labels('data/data.mat', 'data/label.mat')
    gmm = LowRankGMM(k=2, r=100)
    gmm.fit(data)
    print(gmm.plot_likelihoods())