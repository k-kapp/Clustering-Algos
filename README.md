# Clustering-Algos
A bunch of unsupervised clustering algorithms I implemented

I have the following clustering algorithms in this repository:

## OPTICS and DBSCAN

These two algorithms are related in that they create identify clusters based on distances between points. Clusters are successively grown until all points that have a certain density of other points around them, are found in the cluster. OPTICS is an improved version of DBSCAN, in that it deals better with data where density varies between clusters. For further reading, check wikipedia at https://en.wikipedia.org/wiki/OPTICS_algorithm and https://en.wikipedia.org/wiki/DBSCAN.

## Self-organising map (SMO)

An SMO is a neural network that aims to represent data of arbitrary dimensionality in a two-dimensional grid, while preserving topological properties of the data (i.e. if cluster 1 is closer to cluster 2 than cluster 3, then it will also be evident in the 2-d grid). For further reading, please consult https://en.wikipedia.org/wiki/Self-organizing_map.

Note that these algorithms above have the benefit of not requiring the user to specify how many clusters he/she wishes to obtain, unlike the K-means/mediods algorithms, for example.

More code examples coming soon...
