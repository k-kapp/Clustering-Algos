# Clustering-Algos
A bunch of clustering (unsupervised) and classification (supervised) algorithms.

Note that I wrote this code quite a while ago, and haven't checked it thoroughly. Will find some better examples to apply the data to soon. The data the algorithms are applied to here is all simulated.

I have the following clustering algorithms in this repository:

## OPTICS and DBSCAN

These two algorithms are related in that they create identify clusters based on distances between points. Clusters are successively grown until all points that have a certain density of other points around them, are found in the cluster. OPTICS is an improved version of DBSCAN, in that it deals better with data where density varies between clusters. For further reading, check wikipedia at https://en.wikipedia.org/wiki/OPTICS_algorithm and https://en.wikipedia.org/wiki/DBSCAN.

## Self-organising map (SOM)

An SOM is a neural network that aims to represent data of arbitrary dimensionality in a two-dimensional grid, while preserving topological properties of the data (i.e. if cluster 1 is closer to cluster 2 than cluster 3, then it will also be evident in the 2-d grid). For further reading, please consult https://en.wikipedia.org/wiki/Self-organizing_map.

Note that these unsupervised algorithms above have the benefit of not requiring the user to specify how many clusters he/she wishes to obtain, unlike the K-means/mediods algorithms, for example.

## Learning Vector Quantization (LVQ)

A supervised neural networks-based learning algorithm: https://en.wikipedia.org/wiki/Learning_vector_quantization.

More code examples coming soon...
