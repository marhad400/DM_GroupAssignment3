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
def intializeCentroid(data, k, columns, seed=42):
    # Convert to DataFrame if data is a list but not already a DataFrame
    if not isinstance(data, pd.DataFrame):
        if isinstance(data[0], tuple) and len(data[0]) == len(columns):  
            # RGB pixel data case
            data = pd.DataFrame(data, columns=columns)
        elif isinstance(data[0], list) or isinstance(data[0], np.ndarray):
            # General data case
            data = pd.DataFrame(data, columns=[f"x{i}" for i in range(len(data[0]))])
        else:
            raise ValueError("Unsupported data format for initializing centroids.")

    # Randomly select k rows as initial centroids
    random_indices = random.sample(range(len(data)), k)
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
    # Check if columns contain integers (index-based) or strings (column names)
    if isinstance(columns[0], int):  # If integer indices are provided
        for cluster in range(k):
            # Use .iloc with column indices
            cluster_points = data[data['Cluster'] == cluster].iloc[:, columns]
            new_centroid = cluster_points.mean()
            new_centroids.append(new_centroid.tolist())
    else:  # If column names are provided as strings
        for cluster in range(k):
            # Use standard column access with column names
            cluster_points = data[data['Cluster'] == cluster][columns]
            new_centroid = cluster_points.mean()
            new_centroids.append(new_centroid.tolist())
    return new_centroids
