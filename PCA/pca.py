import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

with open('Normalisation.json', 'r') as file:
    data = json.load(file)

feature_matrix = [list(record["vector1"].values()) for record in data]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_matrix)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

pca_data = [{"index": i, "pca_components": list(X_pca[i])} for i in range(len(X_pca))]
with open('pca.json', 'w') as pca_file:
    json.dump(pca_data, pca_file, indent=4)

print("PCA-transformed data saved to pca.json.")
