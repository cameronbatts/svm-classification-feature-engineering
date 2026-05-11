"""
Utilities.
Collect ugly code in this module rather than polluting the notebooks.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans
)
from sklearn.metrics import (
    adjusted_mutual_info_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
    mean_squared_error
)
from sklearn.preprocessing import StandardScaler


def fit_polynomial_regression(X, Y, degree=1, estimator=sklearn.linear_model.LinearRegression,
                              estimator_kwargs={}, scale_predictors=False):
    polynomial_features = sklearn.preprocessing.PolynomialFeatures(degree, include_bias=False).fit(X)
    X_polynomial = polynomial_features.transform(X)
    standard_scaler = sklearn.preprocessing.StandardScaler()
    if scale_predictors:
        X_polynomial = standard_scaler.fit_transform(X_polynomial)
    estimator = estimator(**estimator_kwargs).fit(X_polynomial, Y)
    
    return polynomial_features, standard_scaler, scale_predictors, estimator


def evaluate_polynomial_regression(X, Y, polynomial_features, standard_scaler,
                                   scale_predictors, estimator, plot=False,
                                   x_label='x', y_label='y',
                                   y_ceil=1.0, y_floor=0.0):
    
    X_polynomial = polynomial_features.transform(X)
    
    if scale_predictors:
        X_polynomial = standard_scaler.transform(X_polynomial)
    
    predictions = estimator.predict(X_polynomial)
    mse = mean_squared_error(Y, predictions)
    coefs = estimator.coef_
    l1 = np.abs(coefs).sum()

    if plot:
        plt.scatter(X, Y, label='Data', color='blue')

        x_values = np.arange(np.min(X), np.max(X), 0.001).reshape(-1, 1)
        x_values_polynomial = polynomial_features.transform(x_values)
        
        if scale_predictors:
            x_values_polynomial = standard_scaler.transform(x_values_polynomial)

        predictions = np.maximum(y_floor, np.minimum(y_ceil, estimator.predict(x_values_polynomial)))

        plt.plot(x_values, predictions, "k--", label="Polynomial Fit")

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()

    return {'mse': mse, 'l1': l1}


def plot_decision_boundary(classifier, X_test, Y_test, n_points=1000,
                           x_label='x', y_label='y', alpha=1.0):

    cols = sns.color_palette()[:2]
    cols_light = [tuple(map(lambda x: x + (1 - x) * 0.7, col)) for col in cols]
    cmap_light = ListedColormap(cols_light)

    x1_range = X_test[:, 0].max() - X_test[:, 0].min()
    x2_range = X_test[:, 1].max() - X_test[:, 1].min()
    x1_min, x1_max = X_test[:, 0].min() - 0.1 * x1_range, X_test[:, 0].max() + 0.1 * x1_range
    x2_min, x2_max = X_test[:, 1].min() - 0.1 * x2_range, X_test[:, 1].max() + 0.1 * x2_range
    x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, (x1_max - x1_min) / n_points),
                                   np.arange(x2_min, x2_max, (x2_max - x2_min) / n_points))
    z = classifier.predict(np.c_[x1_mesh.ravel(), x2_mesh.ravel()])
    z = z.reshape(x1_mesh.shape)

    plt.figure()
    plt.pcolormesh(x1_mesh, x2_mesh, z, cmap=cmap_light, shading='auto')

    _ = plt.scatter(
        X_test[Y_test < 1.0, 0], X_test[Y_test < 1.0, 1],
        c=np.array(cols[0]).reshape(1, -1),
        edgecolor='k',
        s=20,
        alpha=alpha
    )
    _ = plt.scatter(
        X_test[Y_test > 0.0, 0], X_test[Y_test > 0.0, 1],
        c=np.array(cols[1]).reshape(1, -1),
        edgecolor='k',
        s=20,
        alpha=alpha
    )

    ax = plt.gca()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def plot_dendrogram(model, dims=(12.0, 8.0), grey=False, color_threshold=189):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0] + 2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack(
        [children, distance, no_of_observations]
    ).astype(float)

    # Plot the corresponding dendrogram
    if grey:
        dendrogram(linkage_matrix, no_labels=True, link_color_func=lambda x: "k")
    else:
        dendrogram(linkage_matrix, no_labels=True, color_threshold=color_threshold, above_threshold_color="k")
    
    plt.gca().set(xticklabels=[], yticklabels=[], yticks=[])
    fig = plt.gcf()
    fig.set_size_inches(*dims)


def fit_k_means(data, k, random_state=42, score_function=calinski_harabasz_score,
                true_labels=None, **kwargs):
    """
    Function to fit a k-means model.
    - data is scaled using the StandardScaler.
    - k is the desired number of clusters.
    - score_function is a heuristic function to measure the quality of clustering.
    - true_labels are the true cluster labels (if known) which can be used
      to compute e.g. the adjusted_mutual_info_score.
    - **kwargs: additional key-value pairs passed to the KMeans constructor.
    """
    k_means = KMeans(n_clusters=k, random_state=random_state, **kwargs).fit(StandardScaler().fit_transform(data))
    cluster_labels = ["cluster_%d" % lab for lab in k_means.labels_]
    if score_function in {calinski_harabasz_score, silhouette_score}:
        score = score_function(data, cluster_labels)
    elif score_function == davies_bouldin_score:
        score = -score_function(data, cluster_labels)
    elif score_function == adjusted_mutual_info_score:
        score = score_function(true_labels, cluster_labels)
    else:
        raise ValueError(f'scoring with {score_function} not implemented')
    return {'score': score, 'k_means': k_means}


def tune_k_means(data, k, plot=True, **kwargs):
    """
    Function to "tune" k-means.
    - k is a collection of values for the desired number of clusters.
    - if plot=True, a plot of k vs the Calinski-Harabasz score is produced.
    - **kwargs: additional key-value arguments passed to the `fit_k_means` utility.
    """
    scores = []
    best_model = None
    best_k = None
    for k_ in k:
        k_means = fit_k_means(data, k_, **kwargs)
        if not scores or k_means['score'] > max(scores):
            best_model = k_means['k_means']
            best_k = k_
        scores.append(k_means['score'])
    if plot:
        plt.plot(k, scores)
    return {'k': k, 'scores': scores, 'best_k': best_k, 'best_model': best_model}


def fit_agglomerative_clustering(data, k, score_function=calinski_harabasz_score,
                                 linkage='ward', metric=None, true_labels=None,
                                 **kwargs):
    """
    Function to fit an agglomerative clustering model.
    - data is scaled using the StandardScaler.
    - k is the desired number of clusters.
    - score_function is a heuristic function to measure the quality of clustering.
    - true_labels are the true cluster labels (if known) which can be used
      to compute e.g. the adjusted_mutual_info_score.
    - **kwargs: additional key-value pairs passed to the AgglomerativeClustering
      constructor.
    """
    # Validate linkage and metric compatibility
    if linkage == 'ward':
        if metric is not None and metric != 'euclidean':
            raise ValueError("linkage='ward' only works with metric='euclidean'")
        metric = 'euclidean'  # Set metric explicitly for ward linkage

    # Fit the AgglomerativeClustering model
    agglomerative_clustering = AgglomerativeClustering(
        n_clusters=k,
        linkage=linkage,
        metric=metric,
        **kwargs
    ).fit(StandardScaler().fit_transform(data))

    cluster_labels = agglomerative_clustering.labels_

    # Compute the clustering score
    if score_function in {calinski_harabasz_score, silhouette_score}:
        score = score_function(data, cluster_labels)
    elif score_function == davies_bouldin_score:
        score = -score_function(data, cluster_labels)
    elif score_function == adjusted_mutual_info_score:
        if true_labels is None:
            raise ValueError("true_labels must be provided for adjusted_mutual_info_score")
        score = score_function(true_labels, cluster_labels)
    else:
        raise ValueError(f"Scoring with {score_function} not implemented")

    return {
        'score': score,
        'agglomerative_clustering': agglomerative_clustering
    }


def tune_agglomerative_clustering(data, k, plot=True, **kwargs):
    """
    Function to "tune" agglomerative clustering models.
    - k is a collection of values for the desired number of clusters.
    - if plot=True, a plot of k vs the Calinski-Harabasz score is produced.
    - **kwargs: additional key-value arguments passed to the `fit_agglomerative_clustering` utility.
    """
    scores = []
    best_model = None
    best_k = None
    for k_ in k:
        agglomerative_clustering = fit_agglomerative_clustering(data, k_, **kwargs)
        if (len(scores) < 1) or (agglomerative_clustering.get('score') > max(scores)):
            best_model = agglomerative_clustering.get('agglomerative_clustering')
            best_k = k_
        scores.append(agglomerative_clustering.get('score'))
    if plot:
        plt.plot(k, scores)
    out = {
        'k': k,
        'scores': scores,
        'best_k': best_k,
        'best_model': best_model
    }
    return out


def fit_dbscan(data, eps, score_function=calinski_harabasz_score, true_labels=None, **kwargs):
    """
    Function to fit a DBSCAN clustering model.
    - data is scaled using the StandardScaler.
    - eps is the maximum distance between two samples
      for them to be considered in the same neighborhood.
    - score_function is a heuristic function to measure the quality of clustering.
    - true_labels are the true cluster labels (if known) which can be used
      to compute e.g. the adjusted_mutual_info_score.
    - **kwargs: additional key-value pairs passed to the DBSCAN
      constructor.
    """
    dbscan = DBSCAN(eps=eps, **kwargs).fit(StandardScaler().fit_transform(data))
    cluster_labels = ["cluster_%d" % lab for lab in dbscan.labels_]
    if score_function in {calinski_harabasz_score, silhouette_score}:
        try:
            score = score_function(
                data[dbscan.labels_ >= 0],
                dbscan.labels_[dbscan.labels_ >= 0]
            ) * np.mean(dbscan.labels_ >= 0) * (1 + np.log(dbscan.labels_.max()))
        except ValueError:
            score = 0
    elif score_function == davies_bouldin_score:
        score = -score_function(
            data[dbscan.labels_ >= 0],
            dbscan.labels_[dbscan.labels_ >= 0]
        )
    elif score_function == adjusted_mutual_info_score:
        score = score_function(true_labels, dbscan.labels_)
    else:
        raise ValueError(f'scoring with {score_function} not implemented')
    return {'score': score, 'dbscan': dbscan}


def tune_dbscan(data, eps, plot=True, **kwargs):

    """

    Function to "tune" DBSCAN models.

    - eps is a collection of epsilon values for DBSCAN.

    - if plot=True, a plot of eps vs the Calinski-Harabasz score is produced.

    - **kwargs: additional key-value arguments passed to the `fit_dbscan` utility.

    """

    scores = []
    best_model = None
    best_eps = None
    for eps_ in eps:
        dbscan = fit_dbscan(data, eps_, **kwargs)
        if not scores or dbscan['score'] > max(scores):
            best_model = dbscan['dbscan']
            best_eps = eps_
        scores.append(dbscan['score'])
    if plot:
        plt.plot(eps, scores)
    return {'eps': eps, 'scores': scores, 'best_eps': best_eps, 'best_model': best_model}
