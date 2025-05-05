import numpy as np
from sklearn import svm
from scripts.save_csv import results_to_csv

# load all the data from MNIST and spam
fields = "test_data", "training_data", "training_labels"
mnist_data = np.load("data/mnist-data.npz")
mnist_test_data = mnist_data[fields[0]]
mnist_train_val_data = mnist_data[fields[1]]
mnist_train_val_labels = mnist_data[fields[2]]
spam_data = np.load("data/spam-data.npz")
spam_test_data = spam_data[fields[0]]
spam_train_val_data = spam_data[fields[1]]
spam_train_val_labels = spam_data[fields[2]]

# For the MNIST dataset, write code that sets aside 10,000 training images as a validation set.
mnist_val_data_num = 10000
mnist_indices = np.random.permutation(len(mnist_train_val_data))
mnist_val_data = mnist_train_val_data[mnist_indices[:mnist_val_data_num]]
mnist_val_labels = mnist_train_val_labels[mnist_indices[:mnist_val_data_num]]
mnist_train_data = mnist_train_val_data[mnist_indices[mnist_val_data_num:]]
mnist_train_labels = mnist_train_val_labels[mnist_indices[mnist_val_data_num:]]
# train the model with 0~train_set_size subset of all the train data
train_set_size = 12000 # use 12,000 samples as train subset
                       # since num > 5000 does not make the model better
                       # from question 4a.
mnist_train_data_subset = mnist_train_data[:train_set_size]
mnist_train_labels_subset = mnist_train_labels[:train_set_size]
# In MNIST, our feature vector for an image will be a row vector 
# with all the pixel values concatenated in a row major (or column major) order
flattened_mnist_train_data_subset = \
    mnist_train_data_subset.reshape((train_set_size, -1))
flattened_mnist_val_data = mnist_val_data.reshape((mnist_val_data_num, -1))
flattened_mnist_test_data = mnist_test_data.reshape((len(mnist_test_data), -1))

# For spam dataset, write code that sets aside 20% of the training data as a validation set.
spam_val_data_split_ratio = 0.2
spam_val_data_num = int(len(spam_train_val_data) * spam_val_data_split_ratio)
spam_indices = np.random.permutation(len(spam_train_val_data))
spam_val_data = spam_train_val_data[spam_indices[:spam_val_data_num]]
spam_val_labels = spam_train_val_labels[spam_indices[:spam_val_data_num]]
spam_train_data = spam_train_val_data[spam_indices[spam_val_data_num:]]
spam_train_labels = spam_train_val_labels[spam_indices[spam_val_data_num:]]

# build the best model for MNIST dataset
mnist_hyper_c = 1.e-6 # best hyper C from question 5
mnist_model = svm.SVC(kernel="linear", max_iter=50000, C=mnist_hyper_c)
mnist_model.fit(flattened_mnist_train_data_subset, mnist_train_labels_subset)
pred_mnist_test_labels = mnist_model.predict(flattened_mnist_test_data)

results_to_csv(pred_mnist_test_labels, "submission_mnist.csv")

# build the best model for spam dataset
spam_hyper_c = 1.e1 # best hyper C from question 6
# After several attempts, I found that the "rbf"-kernel SVM has a better effect
# than the linear-SVM
spam_model = svm.SVC(kernel="rbf", max_iter=10000, C=spam_hyper_c)
spam_model.fit(spam_train_data, spam_train_labels)
pred_spam_test_labels = spam_model.predict(spam_test_data)

results_to_csv(pred_spam_test_labels, "submission_spam.csv")