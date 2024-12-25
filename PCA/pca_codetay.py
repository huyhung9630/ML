import json
import numpy as np

def mean_center(data):
    """Mean center the dataset."""
    mean = np.mean(data, axis=0)
    return data - mean, mean

def compute_covariance_matrix(data):
    """Compute the covariance matrix."""
    return np.cov(data, rowvar=False)

def compute_eigen_decomposition(cov_matrix):
    """Compute eigenvalues and eigenvectors."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    return eigenvalues, eigenvectors

def project_data(data, eigenvectors, num_components):
    """Project the data onto the top eigenvectors."""
    return np.dot(data, eigenvectors[:, :num_components])

with open('Normalisation.json', 'r') as file:
    data = json.load(file)

feature_matrix = np.array([list(record["vector1"].values()) for record in data])

mean_centered_data, mean = mean_center(feature_matrix)

cov_matrix = compute_covariance_matrix(mean_centered_data)

eigenvalues, eigenvectors = compute_eigen_decomposition(cov_matrix)

explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
num_components = np.argmax(cumulative_explained_variance >= 0.90) + 1

projected_data = project_data(mean_centered_data, eigenvectors, num_components)

pca_data = [{"index": i, "pca_components": list(projected_data[i])} for i in range(len(projected_data))]
with open('pca2.json', 'w') as pca_file:
    json.dump(pca_data, pca_file, indent=4)

print(f"PCA with {num_components} components.")
