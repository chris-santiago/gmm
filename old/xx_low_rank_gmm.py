import scipy.io
import numpy as np
import scipy.stats
import scipy.linalg
import matplotlib.pyplot as plt
from sklearn import datasets


def load_data_labels(data_file: str, label_file: str):
    data_contents = scipy.io.loadmat(data_file)
    label_contents = scipy.io.loadmat(label_file)
    return data_contents['data'].T, label_contents['trueLabel'].T


class LowRankGMM:
    def __init__(self, k, r, tol=1e-4, max_iter=100):
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

    def init_means(self, data):
        self.means = np.random.normal(size=(self.K, data.shape[1]))

    def init_covs(self, data):
        self.covs = np.repeat(np.identity(data.shape[1])[np.newaxis, :, :], self.K, axis=0)

    def init_weights(self):
        self.weights = np.repeat(1 / self.K, self.K)

    def init_resp(self, data):
        self.resp = np.full((data.shape[0], self.K), 1 / self.K)

    def initialize(self, data):
        self.init_means(data)
        self.init_covs(data)
        self.init_weights()
        self.init_resp(data)

    def normalize_resp(self):
        self.resp = self.resp / self.resp.sum(axis=1, keepdims=1)

    def e_step(self, data):
        for k in range(self.K):
            eigvals, eigvecs = self.get_r_eigvals_eigvecs(self.covs[k], self.R)
            sigma_approx = eigvecs @ np.diag(eigvals) @ eigvecs.T
            data_approx = data @ eigvecs
            mu_approx = self.means[k] @ eigvecs
            self.resp[:, k] = self.weights[k] * (1 / np.sqrt(np.prod(eigvals))) * np.exp(
                (-1 / 2) * np.sum(((data_approx - mu_approx) ** 2) / eigvals, axis=1))
        self.normalize_resp()

    def m_step(self, data):
        for k in range(self.K):
            self.means[k, :] = np.dot(self.resp[:, k], data) / self.resp[:, k].sum()
            res = []
            for i in range(data.shape[0]):
                res.append(
                    self.resp[i, k] * (data[i, :] - self.means[k]) * (data[i, :] - self.means[k]).reshape(-1, 1))
            self.covs[k] = np.array(res).sum(0) / self.resp[:, k].sum()
            self.weights[k] = self.resp[:, k].sum(0) / data.shape[0]

    def get_loglikelihood(self):
        return np.sum(np.log(np.sum(self.resp, axis=1)))

    def update_loglikelihood(self):
        self.likelihoods.append(self.get_loglikelihood())

    def is_converged(self, curr_iter):
        return abs(self.likelihoods[curr_iter] - self.likelihoods[curr_iter - 1]) < self.tol

    def fit(self, data):
        self.initialize(data)
        curr_iter = 0
        converged = False
        while not converged and curr_iter < self.max_iter:
            self.e_step(data)
            self.m_step(data)
            self.update_loglikelihood()
            if curr_iter > 1:
                converged = self.is_converged(curr_iter)
            curr_iter += 1

    def plot_likelihoods(self):
        plt.plot(self.likelihoods)
        plt.title('Log Likelihoods')
        plt.xlabel('Iteration')
        plt.ylabel('Log Likelihood')
        plt.show()


def main_iris():
    iris = datasets.load_iris()
    data = iris.data
    labels = iris.target

    gmm = LowRankGMM(k=3, r=3)
    gmm.fit(data)
    print(gmm.plot_likelihoods())


def main():
    data, labels = load_data_labels('../../data/data.mat', '../data/label.mat')
    gmm = LowRankGMM(k=2, r=100)
    gmm.fit(data)
    print(gmm.plot_likelihoods())

if __name__ == '__main__':
    main_iris()


