import scipy.io
import numpy as np
import scipy.stats
import scipy.linalg
import matplotlib.pyplot as plt
from sklearn import datasets
from tqdm import tqdm


def load_data_labels(data_file: str, label_file: str):
    data_contents = scipy.io.loadmat(data_file)
    label_contents = scipy.io.loadmat(label_file)
    return data_contents['data'].T, label_contents['trueLabel'].T


class BaseGMM:
    def __init__(self, k, tol=1e-4, max_iter=100):
        self.K = k
        self.tol = tol
        self.max_iter = max_iter

        self.means = None
        self.covs = None
        self.weights = None
        self.resp = None
        self.likelihoods = []

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
            self.resp[:, k] = self.weights[k] * scipy.stats.multivariate_normal.pdf(
                data,
                mean=self.means[k],
                cov=self.covs[k]
            )
        self.update_loglikelihood()
        self.normalize_resp()

    def m_step(self, data):
        for k in range(self.K):
            self.means[k, :] = np.dot(self.resp[:, k], data) / self.resp[:, k].sum()
            res = []
            for i in range(data.shape[0]):
                res.append(
                    self.resp[i, k] * (data[i, :] - self.means[k]) * (data[i, :] - self.means[k]).reshape(-1, 1)
                )
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
        pbar = tqdm(total=self.max_iter)
        while not converged and curr_iter < self.max_iter:
            self.e_step(data)
            self.m_step(data)
            # self.update_loglikelihood()
            if curr_iter > 1:
                converged = self.is_converged(curr_iter)
            curr_iter += 1
            pbar.update()
        pbar.close()

    def plot_likelihoods(self):
        plt.plot(self.likelihoods)
        plt.title('Log Likelihoods')
        plt.xlabel('Iteration')
        plt.ylabel('Log Likelihood')
        plt.show()


def main():
    iris = datasets.load_iris()
    data = iris.data
    labels = iris.target
    gmm = BaseGMM(3, tol=1e-4)
    gmm.fit(data)

if __name__ == '__main__':
    main()
