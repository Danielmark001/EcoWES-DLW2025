import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset (update the file path as needed)
df = pd.read_csv('src/data/waste_collection.csv', delimiter=';')

# Parse the 'geo_point_2d' column to extract latitude and longitude
def parse_geo_point(geo_str):
    try:
        lat_str, lon_str = geo_str.split(',')
        return float(lat_str.strip()), float(lon_str.strip())
    except Exception as e:
        return np.nan, np.nan

# Apply the parser and create new columns
df[['latitude', 'longitude']] = df['geo_point_2d'].apply(lambda x: pd.Series(parse_geo_point(x)))

# Drop any rows where parsing failed
df.dropna(subset=['latitude', 'longitude'], inplace=True)

# Use the latitude and longitude as features for clustering
X = df[['latitude', 'longitude']].values

# Define the number of clusters (adjust 'k' as needed)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Print cluster centroids (optional)
print("Cluster centroids (latitude, longitude):")
print(kmeans.cluster_centers_)

# Visualize the clusters
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for i in range(k):
    cluster_points = df[df['cluster'] == i]
    plt.scatter(cluster_points['longitude'], cluster_points['latitude'],
                color=colors[i], label=f'Cluster {i}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Waste Collection Clusters')
plt.legend()
plt.show()
