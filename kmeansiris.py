import numpy as np
import tensorflow as tf
from sklearn import datasets 

iris = datasets.load_iris()
irispoints = iris.data[:, :2] 


def input_fn():
  return tf.train.limit_epochs(
      tf.convert_to_tensor(irispoints, dtype=tf.float32), num_epochs=1)

num_clusters = 3

# mini_batch is a type of gradient descent algorithm 
kmeans = tf.contrib.factorization.KMeansClustering(
    num_clusters=num_clusters, use_mini_batch=False)

# iterate to find cluster centers  
num_iterations = 8
previous_centers = None
for i in range(num_iterations):
  kmeans.train(input_fn)
  cluster_centers = kmeans.cluster_centers()
  if previous_centers is not None:
    print('delta:', cluster_centers - previous_centers)
  previous_centers = cluster_centers
  print('score:', kmeans.score(input_fn))
print('cluster centers:', cluster_centers)

# map the input irispoints to their clusters
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
for j, point in enumerate(irispoints):
  cluster_index = cluster_indices[j]
  center = cluster_centers[cluster_index]
  print('point:', point, 'is in cluster', cluster_index, 'centered at', center)

