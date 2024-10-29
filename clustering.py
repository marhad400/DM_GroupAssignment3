import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat

# K-means helper functions 
# Normalization 
def normalizeWeather(weatherDf):
    # Specify the columns to normalize
    columns_to_normalize = ['Position', 'Air Temperature', 'Relative Humidity', 'Air Pressure', 
                            'Track Temperature', 'Wind Speed']
                            

    # Normalize each specified column in place
    for column in columns_to_normalize:
        weatherDf[column] = (weatherDf[column] - weatherDf[column].min()) / (weatherDf[column].max() - weatherDf[column].min())

    return weatherDf

# Centroid initialization
# Select k rows randomly acting as starting centroids
def intializeCentroid(data, k, columns, seed=42):  # seed: making sure the same initial centroids are chosen
    # data: full dataset / k: number of clusters & number of centroids / columns: attributes (features) to cluster
    
    # Selecting random k rows from the dataset as initial centroids
    #random.seed(seed)
    random_indices = random.sample(range(len(data)), k)  # random.sample used to select k random items (picking k rows aka data points as initial centroids)
    centroids = [data.iloc[idx][columns].tolist() for idx in random_indices]

    return centroids

# Calculate Euclidean distance
# reference : https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
def dist(point, center, columns):
    return np.sqrt(sum((point[col] - center[idx]) ** 2 for idx, col in enumerate(columns)))

# Assign each data point to the nearest centroid
def assign_clusters(data, centroids, columns):
    clusters = []
    for _, row in data.iterrows():
        distances = [dist(row, centroid, columns) for centroid in centroids]
        clusters.append(np.argmin(distances))  # Assign to nearest centroid, np.argmin returns index of the minimum value in an array. so distances is list of distances from data point to each centroid
    return clusters

# Calculate new centroids as the mean of assigned points
def calculate_centroids(data, clusters, k, columns):
    new_centroids = []
    for cluster in range(k):
        cluster_points = data[data['Cluster'] == cluster][columns]
        new_centroid = cluster_points.mean()
        new_centroids.append(new_centroid)
    return new_centroids


# DO NOT CHANGE THE FOLLOWING LINE
def kmeans(data, k, columns, centers=None, n=None, eps=None):
# DO NOT CHANGE THE PRECEDING LINE
    # This function has to return a list of k cluster centers (lists of floats of the same length as columns)
    if centers is None:
        centroids = intializeCentroid(data, k, columns)
    else:
        centroids = centers # otherwise, use the provided one (centers should be defined)

    for iteration in range(n):
        # Assign clusters
        data['Cluster'] = assign_clusters(data, centroids, columns)
        
        # Calculate new centroids
        new_centroids = calculate_centroids(data, data['Cluster'], k, columns)

        # Check for convergence...
        centroid_shifts = [np.linalg.norm(np.array(new) - np.array(old)) for new, old in zip(new_centroids, centroids)]
        if max(centroid_shifts) < 1e-4:
            break

        centroids = new_centroids

    return centroids, data['Cluster']



# DO NOT CHANGE THE FOLLOWING LINE
def dbscan(data, columns, eps, min_samples):
# DO NOT CHANGE THE PRECEDING LINE
    # This function has to return a list of cluster centers (lists of floats of the same length as columns)
    def euclidean_distance(point1, point2):
        return math.sqrt(sum((point1[i] - point2[i]) ** 2 for i in range(len(columns))))
    
    
    def get_neighbors(point, data):
        neighbors = []
        for other_point in data:
            if euclidean_distance(point, other_point) <= eps:
                neighbors.append(tuple(other_point))
        return neighbors
    
    # Identify core points
    core_points = [tuple(point) for point in data if len(get_neighbors(point, data)) >= min_samples]
    
    # Expand clusters from core points
    visited = set()
    clusters = []
    
    for core_point in core_points:
        if core_point in visited:
            continue  # Skip if we've already processed this point
        
        # Start a new cluster
        cluster = []
        points_to_visit = [core_point]
        
        while points_to_visit:
            point = points_to_visit.pop()
            if point in visited:
                continue  # Skip points that have already been visited
            
            visited.add(point)
            cluster.append(point)
            
            # Get neighbors and expand if they are also core points
            neighbors = get_neighbors(point, data)
            if len(neighbors) >= min_samples:
                # Only expand from core points
                for neighbor in neighbors:
                    if neighbor not in visited:
                        points_to_visit.append(neighbor)
        
        clusters.append(cluster)  # Add the formed cluster to the list of clusters
    
    # Identify and label noise points
    noise = [tuple(point) for point in data if tuple(point) not in visited]
    
    return clusters, noise

# Functions to calculate DBSCAN sillhouette score
def silhouette_score(clusters):
    
    def euclidean_distance(point1, point2):
        return math.sqrt(sum((point1[i] - point2[i]) ** 2 for i in range(len(clusters[0][0]))))
    
    def calculate_centroid(cluster):
        return np.mean(cluster, axis=0)
    
    silhouette_scores = []
    
    # Calculate centroids of all clusters
    cluster_centroids = [calculate_centroid(cluster) for cluster in clusters]
    for c in cluster_centroids:
        print(type(c))
        print(c)
    
    # Calculate silhouette score for each cluster and each point
    for cluster_idx, cluster in enumerate(clusters):
        for point in cluster:

            # Cohesion value
            cohesion_distances = [
                euclidean_distance(point, other_point) for other_point in cluster if other_point != point
            ]
            a = np.mean(cohesion_distances) if cohesion_distances else 0  # avoid divide by zero if single point cluster

            # Separation value
            separation_distances = [
                euclidean_distance(point, centroid) for idx, centroid in enumerate(cluster_centroids) if idx != cluster_idx
            ]
            b = min(separation_distances) if separation_distances else 0  # avoid if no other clusters
            
            # Silhouette ratio
            if max(a, b) > 0:
                silhouette_score_point = (b - a) / max(a, b)
            else:
                silhouette_score_point = 0  # Handle cases with zero distances
            silhouette_scores.append(silhouette_score_point)
    
    overall_silhouette_score = np.mean(silhouette_scores)
    return overall_silhouette_score


def plot_clusters(clusters, noise):
    colors = plt.cm.get_cmap('tab10', len(clusters) + 1)

    plt.figure(figsize=(8, 6))

    for i, cluster in enumerate(clusters):
        cluster_points = np.array(cluster)
        plt.scatter(cluster_points[:, 1], cluster_points[:, 2], s=50, color=colors(i), label=f'Cluster {i+1}')
    
    if noise:
        noise_points = np.array(noise)
        plt.scatter(noise_points[:, 0], noise_points[:, 1], s=50, color='lightgrey', label='Noise')
    
    plt.title("DBSCAN Clustering Results")
    plt.xlabel("Relative Humidity")
    plt.ylabel("Air Pressure")
    plt.legend()
    plt.show()


# DO NOT CHANGE THE FOLLOWING LINE
def kmedoids(data, k, distance, centers=None, n=None, eps=None):
# DO NOT CHANGE THE PRECEDING LINE
    # This function has to return a list of k cluster centroids (data instances!)
    pass

# Functions to get k-means sillhouette score 
def get_all_clusters(k, clusters, data, columns):
  allClusters = []
  def get_cluster_points():
    cluster = []
    for i in range(len(data)):
      if clusters[i] == k:
        cluster.append([data.loc[i, columns[0]], data.loc[i, columns[1]]])
    return cluster

  for k in range(1,k):
    cluster = get_cluster_points()
    allClusters.append(cluster)
  return allClusters

def calculate_separation_point(point, centroids, columns, cluster, clusterPoints):
  # figure out closest cluster
  centroidDistances = []
  centroidNum = 0
  for i in range(len(centroids) - 1):
      # save the distances from this point to other cluster's centroids 
      distToCentroid = dist(centroids[cluster-1], centroids[i+1], columns)
      centroidDistances.append(distToCentroid)
  # find the minimum distance from this point to the centroid of other clusters 
  closestCentroid = min(centroidDistances)

  # using the number of the centroid, locate its cluster index 
  for j in range(0, len(centroidDistances)):
    if centroidDistances[i] == closestCentroid:
      centroidNum = i

  # use an array to store the distances from this point to all other points   
  averageDistances = []
  for k in range(len(clusterPoints[centroidNum])):
    # save the distancee from this point to a point in the next cluster 
    averageDistances.append(twoDEucDist(point, clusterPoints[centroidNum][k]))
  # return the average distance of the distances in the array   
  return stat.mean(averageDistances)


def calculate_cohesion_point(point, cluster):
    # create an array to store the distances of points in the same cluster
    clusterpts = []
    for i in range(len(cluster)):
        # append the distances to points other than the chosen point
        if cluster[i] != point:  
            clusterpts.append(twoDEucDist(point, cluster[i]) )
    # return the average distance from this point to all other points in the same cluster 
    return stat.mean(clusterpts)


def twoDEucDist(p1, p2):
      # function for two dimensional euclidean distance 
      return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))


def main(): 
# Load the dataset
    df = pd.read_csv('assignment3/weatherIncluded3.csv')
    # Normalize specified columns in place
    normalizedDf = normalizeWeather(df)
    columns = ['Relative Humidity', 'Track Temperature']

    # Set the number of clusters
    k = 6

    # Run the k-means algorithm
    centroids, clusters = kmeans(normalizedDf, k, columns, n=100)
    
    # Save the full DataFrame (with normalized and non-normalized columns) to a new CSV file
    #normalizedDf.to_csv('normalized_weather_with_all_columns.csv', index=False)
    
    # Display the first few rows of the normalized data
    # print("First 5 rows only with normalized values for specified columns:")
    # print(normalizedDf.head())

    print("\n\nK-means... ")
    # Visualize the clusters
    plt.scatter(normalizedDf[columns[0]], normalizedDf[columns[1]], c=clusters, cmap='viridis')
    plt.scatter([c[0] for c in centroids], [c[1] for c in centroids], s=300, c='red', marker='X')  # Plot centroids
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.title(f'K-Means Clustering (k={k})')
    plt.show()

    # Testing initialize centroids with example values, 3 random rows representing the initial positions of the centroids
    initial_centroids = intializeCentroid(normalizedDf, k=k, columns=columns)
    print("\nInitial Centroids:")
    print(initial_centroids)


    # Cluster Evaluation: Intrinsic Sillhouette Score

    # Evaluation 1: K-means clustering 
    # Get all the points assigned to each cluster

    clusterPoints = get_all_clusters(k, clusters, normalizedDf, columns)
    silCoefficients = []
    for i in range(len(clusterPoints)):
      for j in range(len(clusterPoints[i])):
        # Cohesion point 
        cohesionPoint = calculate_cohesion_point(clusterPoints[i][j], clusterPoints[i])
        # Separation point
        separationPoint = calculate_separation_point(clusterPoints[i][j], centroids, columns, i, clusterPoints)
        # Calculate sillhouette coefficient 
        silCoefficient = (separationPoint-cohesionPoint)/(max(cohesionPoint, separationPoint))
        silCoefficients.append(silCoefficient)
    print("\nSillhouette Score for k-means:", stat.mean(silCoefficients))


    data = pd.read_csv('assignment3/weatherIncluded3.csv')
    columns = ['Air Temperature', 'Relative Humidity', 'Air Pressure', 'Track Temperature', 'Wind Speed']
    data_points = data[columns].values.tolist()

    # DBSCAN parameters
    eps = 30
    min_samples = 5

    clusters, noise = dbscan(data_points, columns, eps, min_samples)

    # print("Clusters:")
    # for i, cluster in enumerate(clusters):
    #     print(f"Cluster {i+1}: {cluster}")

    print("\n\nDBSCAN...")
    print("Noise points:")
    print(noise)

    sil_score = silhouette_score(clusters)
    print(f"\nSilhouette Score for DBSCAN: {sil_score}")

    plot_clusters(clusters, noise)

    

if __name__ == '__main__':
    main()