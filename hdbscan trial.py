# import psycopg2
# import pandas as pd
# import hdbscan
# import folium
# from folium.plugins import MarkerCluster
# import matplotlib.pyplot as plt




# # Step 1: Connect to PostgreSQL and Fetch Data
# def fetch_data():
#     conn = psycopg2.connect(
#         dbname="innovatemore",
#         user="postgres",
#         password="abcd",
#         host="127.0.0.1",
#         port=5432
#     )
#     query = "SELECT latitude, longitude FROM public.lat_long;"
#     df = pd.read_sql(query, conn)
#     conn.close() 
#     df = df[(df['latitude'] != 0.0 ) & (df['longitude'] != 0.0 )]
#     return df

# # Step 2: Apply HDBSCAN Clustering
# def cluster_data(df):
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric='haversine')
#     df['cluster'] = clusterer.fit_predict(df[['latitude', 'longitude']])
#     return df

# # Step 3: Visualize Clusters on a Geo Map
# def plot_map(df):
    
#     map_center = [df['latitude'].mean(), df['longitude'].mean()]
#     my_map = folium.Map(location=map_center, zoom_start=10)

#     colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue']
    
#     clusters = df['cluster'].unique()
#     for cluster in clusters:
#         cluster_data = df[df['cluster'] == cluster]
#         color = colors[cluster % len(colors)] if cluster != -1 else "gray"  # Gray for outliers
        
#         marker_cluster = MarkerCluster().add_to(my_map)
#         for _, row in cluster_data.iterrows():
#             folium.Marker(
#                 location=[row['latitude'], row['longitude']],
#                 icon=folium.Icon(color=color),
#                 popup=f"Cluster: {cluster}"
#             ).add_to(marker_cluster)

#     my_map.save("zoning_map.html")
#     print("Map saved as zoning_map.html")

# # Run the script
# df = fetch_data()
# df1 = cluster_data(df)
# # print(df1['cluster'].unique())
# plot_map(df1)



# import psycopg2
# import pandas as pd
# import hdbscan
# import folium
# import matplotlib.pyplot as plt
# import numpy as np

# # Step 1: Connect to PostgreSQL and Fetch Data
# def fetch_data():
#     """
#     Connect to PostgreSQL database and fetch latitude-longitude data.
#     Returns a Pandas DataFrame with the data.
#     """

#     conn = psycopg2.connect(
#         dbname="innovatemore",
#         user="postgres",
#         password="abcd",
#         host="127.0.0.1",
#         port=5432
#     )
#     query = "SELECT latitude, longitude FROM public.lat_long;"
#     df = pd.read_sql(query, conn)
#     conn.close() 
#     df = df[(df['latitude'] != 0.0 ) & (df['longitude'] != 0.0 )]
#     return df

# # Step 2: Convert Lat-Long to Radians
# def preprocess_data(df):
#     df[['latitude', 'longitude']] = np.radians(df[['latitude', 'longitude']])
#     return df

# # Step 3: Apply HDBSCAN Clustering with Outlier Control
# def cluster_data(df):
#     clusterer = hdbscan.HDBSCAN(
#         min_cluster_size=5,  # Prevents merging into 1 cluster
#         min_samples=3,  # Includes more border points in clusters
#         metric='haversine',
#         cluster_selection_method='eom',
#         cluster_selection_epsilon=0.005,  # Reduces outliers
#         allow_single_cluster=False  # Forces multiple clusters
#     )
#     df['cluster'] = clusterer.fit_predict(df[['latitude', 'longitude']])
#     return df


# # Step 5: Assign Colors to Clusters
# def assign_colors(df):
#     unique_clusters = df['cluster'].unique()
#     color_palette = ["red", "blue", "green", "purple", "orange", "darkred", "lightred", 
#                      "beige", "darkblue", "darkgreen", "cadetblue", "pink", "black"]

#     color_map = {cluster: color_palette[i % len(color_palette)] for i, cluster in enumerate(unique_clusters)}
#     color_map[-1] = "gray"  # Outliers get gray color
#     return color_map

# # Step 6: Plot on Geo Map
# def plot_map(df, color_map):
#     map_center = [np.degrees(df['latitude'].mean()), np.degrees(df['longitude'].mean())]
#     my_map = folium.Map(location=map_center, zoom_start=10)

#     for _, row in df.iterrows():
#         folium.CircleMarker(
#             location=[np.degrees(row['latitude']), np.degrees(row['longitude'])], 
#             radius=5, color=color_map[row['cluster']], fill=True, fill_color=color_map[row['cluster']]
#         ).add_to(my_map)

#     my_map.save("zoning_map2.html")
#     print("Map saved as zoning_map.html")

# # Run the script
# df = fetch_data()
# df = preprocess_data(df)
# df = cluster_data(df) 
# print(df['cluster'].unique())
# color_map = assign_colors(df)
# plot_map(df, color_map)




import numpy as np
import psycopg2
import pandas as pd
import hdbscan
import folium
from sklearn.cluster import KMeans

# Step 1: Connect to PostgreSQL and Fetch Data
def fetch_data():
    """
    Connect to PostgreSQL database and fetch latitude-longitude data.
    Returns a Pandas DataFrame with the data.
    """

    conn = psycopg2.connect(
        dbname="innovatemore",
        user="postgres",
        password="abcd",
        host="127.0.0.1",
        port=5432
    )
    query = "SELECT latitude, longitude FROM public.lat_long;"
    df = pd.read_sql(query, conn)
    conn.close() 
    df = df[(df['latitude'] != 0.0 ) & (df['longitude'] != 0.0 )]
    return df

# Step 2: Convert Lat-Long to Radians
def preprocess_data(df):
    df[['latitude', 'longitude']] = np.radians(df[['latitude', 'longitude']])
    return df

# Step 3: Apply HDBSCAN Clustering with Outlier Control
def cluster_data(df):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=4,  # Allows smaller clusters
        min_samples=3,  # Includes more border points in clusters
        metric='haversine',
        cluster_selection_method='eom',
        cluster_selection_epsilon=0.01,  # Helps split large clusters
        allow_single_cluster=False  # Prevents everything from merging into one cluster
    )
    df['cluster'] = clusterer.fit_predict(df[['latitude', 'longitude']])
    return df

# Step 4: Split Large Clusters Using KMeans
def split_large_clusters(df, max_size=7):
    large_clusters = df['cluster'].value_counts()
    large_clusters = large_clusters[large_clusters > max_size].index.tolist()

    for cluster in large_clusters:
        cluster_points = df[df['cluster'] == cluster]
        num_subclusters = len(cluster_points) // max_size

        if num_subclusters > 1:
            kmeans = KMeans(n_clusters=num_subclusters, random_state=42, n_init=10)
            new_labels = kmeans.fit_predict(cluster_points[['latitude', 'longitude']])

            for i, new_label in enumerate(new_labels):
                df.loc[cluster_points.index[i], 'cluster'] = f"{cluster}_{new_label}"  # Rename clusters

    return df

# Step 5: Assign Colors to Clusters
def assign_colors(df):
    unique_clusters = df['cluster'].unique()
    color_palette = ["red", "blue", "green", "purple", "orange", "darkred", "lightred", 
                     "beige", "darkblue", "darkgreen", "cadetblue", "pink", "black"]

    color_map = {cluster: color_palette[i % len(color_palette)] for i, cluster in enumerate(unique_clusters)}
    color_map[-1] = "gray"  # Outliers get gray color
    return color_map

# Step 6: Plot on Geo Map
def plot_map(df, color_map):
    map_center = [np.degrees(df['latitude'].mean()), np.degrees(df['longitude'].mean())]
    my_map = folium.Map(location=map_center, zoom_start=10)

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[np.degrees(row['latitude']), np.degrees(row['longitude'])], 
            radius=5, color=color_map[row['cluster']], fill=True, fill_color=color_map[row['cluster']]
        ).add_to(my_map)

    my_map.save("zoning_map3.html")
    print("Map saved as zoning_map.html")

# Run the script
df = fetch_data()
df = preprocess_data(df)
df = cluster_data(df)
df = split_large_clusters(df, max_size=7)  # Fix big clusters
print(df['cluster'].unique())
color_map = assign_colors(df)
plot_map(df, color_map)
