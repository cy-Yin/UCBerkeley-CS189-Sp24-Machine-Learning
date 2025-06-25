import numpy as np
import scipy
from scripts.save_csv import results_to_csv
np.random.seed(42)

# Load mnist data
mnist_data = np.load((f"data/spam-data-hw3.npz"))
fields = "test_data", "training_data", "training_labels"
test_data = mnist_data[fields[0]]
training_data = mnist_data[fields[1]]
training_labels = mnist_data[fields[2]]

# two Classes: QDA and LDA
class LDA():
    def __init__(self):
        self.means = None
        self.cov = None
        self.labels = None
        self.prior_probs = None

    def fit(self, data, labels):
        n_samples, n_features = data.shape

        self.labels = np.unique(labels)
        self.means = {}
        self.prior_probs = {}
        # cov matrix = 1 / n 
        #       * \sum_{classes}{\sum_{yi = class}{(X_i - mean_class)(X_i - mean_class)^T}}
        self.cov = np.zeros((n_features, n_features))
        
        for label in self.labels:
            data_with_label = data[labels == label]
            # calculate the mean of each label
            mean = np.mean(data_with_label, axis=0)
            self.means[label] = mean
            # calculate the each-label-term of the LDA cov matrix
            diff = data_with_label - mean
            cov = np.dot(diff.T, diff)
            self.cov += cov
            # calculate the prior probability of each class
            prior_prob = data_with_label.shape[0] / n_samples
            self.prior_probs[label] = prior_prob
        self.cov = self.cov / n_samples

    def predict(self, data):
        if self.means is None or self.cov is None:
            raise ValueError("Model not fitted yet!")

        n_samples = data.shape[0]
        n_pred_labels = len(self.labels)

        log_probs = {}
        cov_lda = self.cov
        for label in self.labels:
            mean = self.means[label]
            prior_prob = self.prior_probs[label]
            log_pdf = scipy.stats.multivariate_normal.logpdf(data, 
                                                             mean=mean, cov=cov_lda,
                                                             allow_singular=True) \
                        + np.log(prior_prob)
            log_probs[label] = log_pdf

        all_log_probs = np.column_stack([log_probs[label] for label in self.labels])
        idx_prob_max = np.argmax(all_log_probs, axis=1)
        labels_list = list(self.labels)
        pred_labels = np.array([labels_list[idx] for idx in idx_prob_max])

        return pred_labels
    
class QDA():
    def __init__(self):
        self.means = None
        self.covs = None
        self.labels = None
        self.prior_probs = None

    def fit(self, data, labels):
        n_samples, n_features = data.shape

        self.labels = np.unique(labels)
        self.means = {}
        self.prior_probs = {}
        self.covs = {}
        
        for label in self.labels:
            data_with_label = data[labels == label]
            n_samples_with_label = data_with_label.shape[0]
            # calculate the mean of each label
            mean = np.mean(data_with_label, axis=0)
            self.means[label] = mean
            # calculate the cov matrix of each label
            diff = data_with_label - mean
            cov = np.dot(diff.T, diff) / n_samples_with_label
            # Add the Q7b trick to avoid the singular cov matrix
            # epsilon = 1e-8
            # cov += epsilon * np.eye(cov.shape[0])
            self.covs[label] = cov
            # calculate the prior probability of each class
            prior_prob = n_samples_with_label / n_samples
            self.prior_probs[label] = prior_prob

    def predict(self, data):
        if self.means is None or self.covs is None:
            raise ValueError("Model not fitted yet!")

        n_samples = data.shape[0]
        n_pred_labels = len(self.labels)

        log_probs = {}
        for label in self.labels:
            mean = self.means[label]
            cov = self.covs[label]
            prior_prob = self.prior_probs[label]
            log_pdf = scipy.stats.multivariate_normal.logpdf(data, 
                                                             mean=mean, cov=cov,
                                                             allow_singular=True) \
                        + np.log(prior_prob)
            log_probs[label] = log_pdf

        all_log_probs = np.column_stack([log_probs[label] for label in self.labels])
        idx_prob_max = np.argmax(all_log_probs, axis=1)
        labels_list = list(self.labels)
        pred_labels = np.array([labels_list[idx] for idx in idx_prob_max])

        return pred_labels
    
if __name__ == "__main__":
    # Kaggle SPAM LDA
    lda = LDA()
    lda.fit(training_data, training_labels)
    pred_test_labels = lda.predict(test_data)
    results_to_csv(pred_test_labels)

    # Kaggle SPAM QDA
    qda = QDA()
    qda.fit(training_data, training_labels)
    pred_test_labels = qda.predict(test_data)
    results_to_csv(pred_test_labels)