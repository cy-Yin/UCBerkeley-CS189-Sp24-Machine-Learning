"""
The goal of this assignment is to predict GPS coordinates from image features using k-Nearest Neighbors.
Specifically, have featurized 28616 geo-tagged images taken in Spain split into training and test sets (27.6k and 1k).

The assignment walks students through:
    * visualizing the data
    * implementing and evaluating a kNN regression model
    * analyzing model performance as a function of dataset size
    * comparing kNN against linear regression

Images were filtered from Mousselly-Sergieh et al. 2014 (https://dl.acm.org/doi/10.1145/2557642.2563673)
and scraped from Flickr in 2024. The image features were extracted using CLIP ViT-L/14@336px (https://openai.com/clip/).
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def plot_data(train_feats, train_labels):
    """
    Input:
        train_feats: Training set image features
        train_labels: Training set GPS (lat, lon)

    Output:
        Displays plot of image locations, and first two PCA dimensions vs longitude
    """
    # Plot image locations (use marker='.' for better visibility)
    plt.figure()
    plt.scatter(train_labels[:, 1], train_labels[:, 0], marker=".")
    plt.title('Image Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
    plt.savefig('q4a_image_locations.png')

    # Run PCA on training_feats
    ##### TODO(a): Your Code Here #####
    transformed_feats = StandardScaler().fit_transform(train_feats)
    transformed_feats = PCA(n_components=2).fit_transform(transformed_feats)

    # Plot images by first two PCA dimensions (use marker='.' for better visibility)
    plt.figure()
    plt.scatter(transformed_feats[:, 0],     # Select first column
                transformed_feats[:, 1],     # Select second column
                c=train_labels[:, 1],
                marker='.')
    plt.colorbar(label='Longitude')
    plt.title('Image Features by Longitude after PCA')
    plt.show()
    plt.savefig('q4a_image_features_pca.png')


def grid_search(train_features, train_labels, test_features, test_labels, is_weighted=False, verbose=True):
    """
    Input:
        train_features: Training set image features
        train_labels: Training set GPS (lat, lon) coords
        test_features: Test set image features
        test_labels: Test set GPS (lat, lon) coords
        is_weighted: Weight prediction by distances in feature space

    Output:
        Prints mean displacement error as a function of k
        Plots mean displacement error vs k

    Returns:
        Minimum mean displacement error
    """
    # Evaluate mean displacement error (in miles) of kNN regression for different values of k
    # Technically we are working with spherical coordinates and should be using spherical distances, but within a small
    # region like Spain we can get away with treating the coordinates as cartesian coordinates.
    knn = NearestNeighbors(n_neighbors=100).fit(train_features)

    if verbose:
        print(f'Running grid search for k (is_weighted={is_weighted})')

    ks = list(range(1, 11)) + [20, 30, 40, 50, 100]
    mean_errors = []
    for k in ks:
        distances, indices = knn.kneighbors(test_features, n_neighbors=k)

        errors = []
        for i, nearest in enumerate(indices):
            # Evaluate mean displacement error in miles for each test image
            # Assume 1 degree latitude is 69 miles and 1 degree longitude is 52 miles
            y = test_labels[i]

            ##### TODO(d): Your Code Here #####
            if not is_weighted:
                y_pred = np.mean(train_labels[nearest], axis=0)
                lat_diff = (y[0] - y_pred[0]) * 69
                lon_diff = (y[1] - y_pred[1]) * 52
                e = np.sqrt(lat_diff ** 2 + lon_diff ** 2)
            else:
                dists = distances[i]
                weights = 1 / (dists + 1e-8)  # Avoid division by zero
                weights = weights / np.sum(weights)  # Normalize weights
                y_pred = np.sum(train_labels[nearest] * weights[:, np.newaxis], axis=0)
                lat_diff = (y[0] - y_pred[0]) * 69
                lon_diff = (y[1] - y_pred[1]) * 52
                e = np.sqrt(lat_diff ** 2 + lon_diff ** 2)

            errors.append(e)
        
        e = np.mean(np.array(errors))
        mean_errors.append(e)
        if verbose:
            print(f'{k}-NN mean displacement error (miles): {e:.1f}')

    # Plot error vs k for k Nearest Neighbors
    if verbose:
        plt.figure()
        plt.plot(ks, mean_errors)
        plt.xlabel('k')
        plt.ylabel('Mean Displacement Error (miles)')
        plt.title('Mean Displacement Error (miles) vs. k in kNN')
        plt.show()
        plt.savefig(f'q4{"d" if not is_weighted else "e"}_mean_displacement_error_vs_k{"_weighted" if is_weighted else ""}.png')

    return min(mean_errors)


def main():
    print("Predicting GPS from CLIP image features\n")

    # Import Data
    print("Loading Data")
    data = np.load('im2spain_data.npz')

    train_features = data['train_features']  # [N_train, dim] array
    test_features = data['test_features']    # [N_test, dim] array
    train_labels = data['train_labels']      # [N_train, 2] array of (lat, lon) coords
    test_labels = data['test_labels']        # [N_test, 2] array of (lat, lon) coords
    train_files = data['train_files']        # [N_train] array of strings
    test_files = data['test_files']          # [N_test] array of strings

    # Data Information
    print('Train Data Count:', train_features.shape[0])

    # Part A: Feature and label visualization (modify plot_data method)
    plot_data(train_features, train_labels)

    # Part C: Find the 5 nearest neighbors of test image 53633239060.jpg
    knn = NearestNeighbors(n_neighbors=3).fit(train_features)

    # Use knn to get the k nearest neighbors of the features of image 53633239060.jpg
    ##### TODO(c): Your Code Here #####
    test_image_file = '53633239060.jpg'
    test_image_idx = np.where(test_files == test_image_file)
    distances, indices = knn.kneighbors(test_features[test_image_idx], n_neighbors=3)
    for i in range(3):
        neighbor_idx = indices[0][i] # `distances`, `indices` have shape of (num_queries, num_neighbors). Here we only have one query, so we take `indices[0]`
        neighbor_file = train_files[neighbor_idx]
        print(f'Neighbor {i+1}: {neighbor_file}, Distance: {distances[0][i]:.4f}')
        plt.figure()
        plt.imshow(plt.imread(f'im2spain_images/{neighbor_file}'))
        plt.axis('off')
        plt.title(f'Neighbor {i + 1}: {neighbor_file}')
        plt.savefig(f'q4b_neighbor_{i + 1}_{neighbor_file}.png')

    # Part D: establish a naive baseline of predicting the mean of the training set
    ##### TODO(d): Your Code Here #####
    # simply predicting the training set centroid (coordinate-wise average) location for every test image
    train_centroid = np.mean(train_labels, axis=0)
    lat_diffs = (test_labels[:, 0] - train_centroid[0]) * 69
    lon_diffs = (test_labels[:, 1] - train_centroid[1]) * 52
    e_baseline = np.mean(np.sqrt(lat_diffs ** 2 + lon_diffs ** 2)) # Euclidean distances
    print(f'\nBaseline mean displacement error (miles): {e_baseline:.1f}')

    # Part E: complete grid_search to find the best value of k
    grid_search(train_features, train_labels, test_features, test_labels)

    # Parts G: rerun grid search after modifications to find the best value of k
    grid_search(train_features, train_labels, test_features, test_labels, is_weighted=True)

    # Part H: compare to linear regression for different # of training points
    mean_errors_lin = []
    mean_errors_nn = []
    ratios = np.arange(0.1, 1.1, 0.1)
    for r in ratios:
        num_samples = int(r * len(train_features))
        ##### TODO(h): Your Code Here #####
        # Linear Regression
        linReg = LinearRegression()
        linReg.fit(train_features[:num_samples], train_labels[:num_samples])
        predictions = linReg.predict(test_features)
        lat_diffs = (test_labels[:, 0] - predictions[:, 0]) * 69
        lon_diffs = (test_labels[:, 1] - predictions[:, 1]) * 52
        e_lin = np.mean(np.sqrt(lat_diffs ** 2 + lon_diffs ** 2))
        # kNN
        e_nn = grid_search(train_features[:num_samples], train_labels[:num_samples],
                           test_features, test_labels, is_weighted=True, verbose=False)
        mean_errors_lin.append(e_lin)
        mean_errors_nn.append(e_nn)

        print(f'\nTraining set ratio: {r} ({num_samples})')
        print(f'Linear Regression mean displacement error (miles): {e_lin:.1f}')
        print(f'kNN mean displacement error (miles): {e_nn:.1f}')

    # Plot error vs training set size
    plt.figure()
    plt.plot(ratios, mean_errors_lin, label='lin. reg.')
    plt.plot(ratios, mean_errors_nn, label='kNN')
    plt.xlabel('Training Set Ratio')
    plt.ylabel('Mean Displacement Error (miles)')
    plt.title('Mean Displacement Error (miles) vs. Training Set Ratio')
    plt.legend()
    plt.show()
    plt.savefig('q4g_MDE_vs_training_set_ratio_linReg_vs_kNN.png')
       

if __name__ == '__main__':
    main()
