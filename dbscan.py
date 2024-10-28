import math
import numpy as np

def dbscan(data, columns, eps, min_samples):

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