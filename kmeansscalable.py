import tensorflow as tf
import numpy as np
from randomclusters import create_points_in_clusters

n_features = 2
n_clusters = 3
n_samples_per_cluster = 400
seed = 750
spread_factor = 100

np.random.seed(seed)

input_samples = create_points_in_clusters(n_features, n_clusters, n_samples_per_cluster, seed, spread_factor)

print("input_samples =", input_samples)

# shuffle generated input points so they are not stored in order 
np.random.shuffle(input_samples) 

def input_fn():
  return tf.train.limit_epochs(
      tf.convert_to_tensor(input_samples, dtype=tf.float32), num_epochs=1)

kmeans = tf.contrib.factorization.KMeansClustering(
    num_clusters=n_clusters, relative_tolerance=0.0001, use_mini_batch=True,mini_batch_steps_per_iteration=1, random_seed=2)

# model training 
num_iterations = 9
previous_centers = None
for i in range(num_iterations):
  kmeans.train(input_fn)
  cluster_centers = kmeans.cluster_centers()
  if previous_centers is not None:
    print('delta:', cluster_centers - previous_centers)
  previous_centers = cluster_centers
  print('score:', kmeans.score(input_fn))
print('cluster centers:', cluster_centers)

# map randomly generated input points to their clusters
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
for j, point in enumerate(input_samples):
  cluster_index = cluster_indices[j]
  center = cluster_centers[cluster_index]
  print('point:', point, 'is in cluster', cluster_index, 'with centroid', center)

