import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize

with open('pca.json', 'r') as file:
    pca_data = json.load(file)

X_pca = np.array([record["pca_components"] for record in pca_data])
X_pca = normalize(X_pca)

optimal_k = 16

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_pca)

for i, record in enumerate(pca_data):
    record['cluster'] = int(kmeans.labels_[i])

cluster_stats = {i: 0 for i in range(optimal_k)}
for label in kmeans.labels_:
    cluster_stats[label] += 1

with open('kmeans_results.txt', 'w') as result_file:
    result_file.write("Cluster Statistics:\n")
    for cluster, count in cluster_stats.items():
        result_file.write(f"Cluster {cluster}: {count} points\n")
    
    result_file.write("\nPoint Assignments:\n")
    for record in pca_data:
        result_file.write(f"Point {record['index']} belongs to Cluster {record['cluster']}\n")

print("Done")
