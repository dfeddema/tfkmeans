import tensorflow as tf
import numpy as np

def create_points_in_clusters(n_features, n_clusters, n_samples_per_cluster, seed, spread_factor):
      np.random.seed(seed)
      matrices = []
      centroids = []
# Create blob of points around each centroid 
      for i in range(n_clusters):
          samples = np.random.normal(loc=0,scale=0.5,size=(n_samples_per_cluster, n_features)) 
          current_centroid = (np.random.random((1, n_features)) * spread_factor) - (spread_factor/2)
          centroids.append(current_centroid)
# add the centroid value to each sample point so that samples are an offset from centroid
          samples += current_centroid
          matrices.append(samples)
# contatentate all points for each centroid into "allpoints" dataset
      allpoints  = np.concatenate(matrices,axis=0)
      return allpoints 
