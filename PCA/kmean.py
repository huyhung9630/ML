import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize

# Load the data from Normalisation.json
with open('Normalisation.json', 'r') as file:
    data = json.load(file)

# Extract the 'vector1' field and create a feature matrix
feature_matrix = [list(record["vector1"].values()) for record in data]
feature_matrix = normalize(feature_matrix)

# Determine optimal K using the Elbow Method (optional visualization, not required here)
optimal_k = 16

# Apply K-Means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(feature_matrix)

# Add cluster labels to the data
for i, record in enumerate(data):
    record['cluster'] = int(kmeans.labels_[i])

# Calculate cluster statistics
cluster_stats = {i: 0 for i in range(optimal_k)}
for label in kmeans.labels_:
    cluster_stats[label] += 1

# Write the results to a text file
with open('kmean_normal.txt', 'w') as result_file:
    result_file.write("Cluster Statistics:\n")
    for cluster, count in cluster_stats.items():
        result_file.write(f"Cluster {cluster}: {count} points\n")
    
    result_file.write("\nPoint:\n")
    for record in data:
        result_file.write(f"Point {record['title']} : Cluster {record['cluster']}\n")

print("Done")
