import statistics as stat
import numpy as np 
import matplotlib.pyplot as plt 
import math 
import kmeansHelper
import pandas as pd 
import clustering


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
      distToCentroid = twoDEucDist(centroids[cluster-1], centroids[i+1])
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

