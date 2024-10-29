import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
import kmeansHelper as helper
import analysis

SHOW_STEPS = True  # global variable in order to control steps visualization
ROUND_CENTROIDS = False  # global variable in order to control centroid rounding


# DO NOT CHANGE THE FOLLOWING LINE
def lloyds(data, k, columns, centers=None, n=None, eps=None):
# DO NOT CHANGE THE PRECEDING LINE
    if n is None:
        n = 100

    # Check if data is a list and convert to DataFrame if needed
    if isinstance(data, list):
        data = pd.DataFrame(data, columns=[f"x{i}" for i in range(len(data[0]))])

    if centers is None:
        centroids = helper.intializeCentroid(data, k, columns)
    else:
        centroids = centers  # Use the provided centers if defined

    for iteration in range(n):
        # Assign clusters
        data['Cluster'] = helper.assign_clusters(data, centroids, columns)
        new_centroids = helper.calculate_centroids(data, data['Cluster'], k, columns)

        # Show steps only if SHOW_STEPS is True
        if SHOW_STEPS:
            plt.figure()
            if isinstance(columns[0], int):
                plt.scatter(data.iloc[:, columns[0]], data.iloc[:, columns[1]], c=data['Cluster'], cmap='viridis')
            else:
                plt.scatter(data[columns[0]], data[columns[1]], c=data['Cluster'], cmap='viridis')
            plt.scatter([c[0] for c in new_centroids], [c[1] for c in new_centroids], s=300, c='red', marker='X')
            plt.xlabel(columns[0])
            plt.ylabel(columns[1])
            plt.title(f'Iteration {iteration + 1}')
            plt.show()

        centroid_shifts = [np.linalg.norm(np.array(new) - np.array(old)) for new, old in zip(new_centroids, centroids)]
        if max(centroid_shifts) < 1e-4:
            break

        centroids = new_centroids

    # Final rounding of centroids if ROUND_CENTROIDS is True
    if ROUND_CENTROIDS:
        centroids = [[round(val) if not np.isnan(val) else 0 for val in centroid] for centroid in centroids]
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



# DO NOT CHANGE THE FOLLOWING LINE
def kmedoids(data, k, distance, centers=None, n=None, eps=None):
# DO NOT CHANGE THE PRECEDING LINE
    # This function has to return a list of k cluster centroids (data instances!)
    pass


def main(): 
# Load the dataset
    df = pd.read_csv('weatherIncluded3.csv')
    # Normalize specified columns in place
    normalizedDf = helper.normalizeWeather(df)
    columns = ['Relative Humidity', 'Track Temperature']

    # Set the number of clusters
    k = 6

    # Run the k-means algorithm
    global centroids
    centroids, clusters = lloyds(normalizedDf, k, columns, n=100)
    

    # Save the full DataFrame (with normalized and non-normalized columns) to a new CSV file
    normalizedDf.to_csv('normalized_weather_with_all_columns.csv', index=False)
    

    print("\n\nClustering With K-Means")
    # Visualize the clusters
    plt.scatter(normalizedDf[columns[0]], normalizedDf[columns[1]], c=clusters, cmap='viridis')
    plt.scatter([c[0] for c in centroids], [c[1] for c in centroids], s=300, c='red', marker='X')  # Plot centroids
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.title(f'K-Means Clustering (k={k})')
    plt.show()

    # Testing initialize centroids with example values, 3 random rows representing the initial positions of the centroids
    initial_centroids = helper.intializeCentroid(normalizedDf, k=k, columns=columns)
    print("Initial Centroids:")
    print(initial_centroids)

    data = pd.read_csv('weatherIncluded3.csv')
    columns = ['Air Temperature', 'Relative Humidity', 'Air Pressure', 'Track Temperature', 'Wind Speed']
    data_points = data[columns].values.tolist()

    # Cluster Evaluation: Intrinsic Sillhouette Score
    # Evaluation 1: K-means clustering 
    # Get all the points assigned to each cluster
    df = pd.read_csv("weatherIncluded3.csv")
    normalizedDf = helper.normalizeWeather(df)
    k = 6
    clusterPoints = analysis.get_all_clusters(k, clusters, normalizedDf, columns)
    silCoefficients = []
    for i in range(len(clusterPoints)):
      for j in range(len(clusterPoints[i])):
        # Cohesion point 
        cohesionPoint = analysis.calculate_cohesion_point(clusterPoints[i][j], clusterPoints[i])
        # Separation point
        separationPoint = analysis.calculate_separation_point(clusterPoints[i][j], centroids, columns, i, clusterPoints)
        # Calculate sillhouette coefficient 
        silCoefficient = (separationPoint-cohesionPoint)/(max(cohesionPoint, separationPoint))
        silCoefficients.append(silCoefficient)
    print("\nSillhouette Score for k-means:", stat.mean(silCoefficients))


    data = pd.read_csv('weatherIncluded3.csv')
    columns = ['Air Temperature', 'Relative Humidity', 'Air Pressure', 'Track Temperature', 'Wind Speed']
    data_points = data[columns].values.tolist()

    # DBSCAN parameters
    eps = 30
    min_samples = 5

    clusters, noise = dbscan(data_points, columns, eps, min_samples)

    print("\n\nDBSCAN...")
    print("Noise points:")
    print(noise)

    sil_score = analysis.silhouette_score(clusters)
    print(f"\nSilhouette Score for DBSCAN: {sil_score}")

    analysis.plot_clusters(clusters, noise)
    

if __name__ == '__main__':
    main()
