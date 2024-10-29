import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dbscan import dbscan

def main():

    df = pd.read_csv('weatherIncluded3.csv')

    df = rain_to_numerical(df)
    df = teams_to_numerical(df)

    columns = ['Air Temperature', 'Relative Humidity', 'Air Pressure', 'Track Temperature', 'Wind Speed', 'Rainfall', 'Race Point', 'Driver Team']
    data_points = df[columns].values.tolist()

    # DBSCAN parameters
    eps = 20
    min_samples = 5

    clusters, noise = dbscan(data_points, columns, eps, min_samples)

    # print("Clusters:")
    # for i, cluster in enumerate(clusters):
    #     print(f"Cluster {i+1}: {cluster}")

    print("\nNoise points:")
    print(noise)

    sil_score = silhouette_score(clusters)
    print(f"Silhouette Score: {sil_score}")

    x_feature_index = -1
    y_feature_index = -2
    x_label = columns[x_feature_index]
    y_label = columns[y_feature_index]

    plot_clusters(clusters, noise, x_feature_index, y_feature_index, 
                x_label=x_label, y_label=y_label)


def rain_to_numerical(df):

    rain_value_dict = {True: 1, False: 0, "True": 1, "False": 0}
    df['Rainfall'] = df['Rainfall'].map(rain_value_dict)

    return df

def teams_to_numerical(df):
    
    teams = df['Driver Team'].unique().tolist()

    teams_dict = {item: i+1 for i, item in enumerate(teams)}
    print(f'Teams: {teams_dict}')

    df['Driver Team'] = df['Driver Team'].map(teams_dict)

    return df


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


def plot_clusters(clusters, noise, x_feature_index, y_feature_index, x_label, y_label):
    colors = plt.cm.get_cmap('tab10', len(clusters) + 1)

    plt.figure(figsize=(8, 6))

    for i, cluster in enumerate(clusters):
        cluster_points = np.array(cluster)
        plt.scatter(cluster_points[:, x_feature_index], cluster_points[:, y_feature_index], 
                    s=50, color=colors(i), label=f'Cluster {i+1}')
    
    if noise:
        noise_points = np.array(noise)
        plt.scatter(noise_points[:, x_feature_index], noise_points[:, y_feature_index], 
                    s=50, color='lightgrey', label='Noise')
    
    plt.title("DBSCAN Clustering Results")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()