# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:44:10 2015

@author: Konrad
"""

import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

def euclid(pos1, pos2):
    return np.sqrt(sum((pos1 - pos2)**2))
    
def make_pts(data):
    pts = []
    for pos in data:
        pts.append(Point(pos))
    return pts
    
def gen_clusters(means, num_each):
    tup = ()
    for m in means:
        tup = tup + (np.random.multivariate_normal(m, np.diag(np.ones(2)), num_each),)
    data = np.concatenate(tup)
    np.random.shuffle(data)
    return data

class K_nearest:
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.nearest_k = self.get_all()
        
    def get_k_nearest(self, idx):
        lst = [[0,float("inf")] for i in range(self.k)]
        for i, obj in enumerate(self.data):
            if not (i == idx):
                self.insert(lst, i, euclid(obj.pos, self.data[idx].pos))
        return lst[-1][1]
        
    def get_all(self):
        all_dist = []
        for idx in range(len(data)):
            all_dist.append(self.get_k_nearest(idx))
        return sorted(all_dist, reverse=True)

    def insert(self, lst, cand_idx, dist):
        inserted = False
        for i, obj in enumerate(lst):
            if dist < obj[1]:
                lst.insert(i, [cand_idx, dist])
                inserted = True
                break
        if (inserted):
            del lst[-1]

class State(Enum):
    Proc = 0
    UnProc = 1

class Point:
    def __init__(self, pos):
        self.pos = pos
        self.id = None
        self.core = False

class DBSCAN:
    def __init__(self, min_pts, eps, data):
        self.min_pts = min_pts
        self.eps = eps
        self.data = data
        self.clId = [0]
        
    def main_loop(self):
        for idx in range(len(self.data)):
            if (self.data[idx].id == None):
                self.expand(idx)
                
            
    def expand(self, idx):
        proc_list = self.get_neighbours(idx)
        
        if len(proc_list) < self.min_pts:
            self.data[idx].id = -1
            return False
        
        # From here on, we know that all neighbours are part of this cluster,
        # and that this point (idx) is a core point

        self.clId.append(self.clId[-1] + 1)
        self.data[idx].core = True
        self.data[idx].id = self.clId[-1]
        
        for n_idx in proc_list:
            self.data[n_idx].id = self.clId[-1]
        
        while not (len(proc_list) == 0):
            n_idx = proc_list[0]
            del proc_list[0]
            new_nbrs = self.get_neighbours(n_idx)
            if (len(new_nbrs) >= self.min_pts):
                self.data[n_idx].core = True
                for new_n_idx in new_nbrs:
                    if self.data[new_n_idx].id == None or self.data[new_n_idx].id == -1:
                        if self.data[new_n_idx].id == None:
                            proc_list.append(new_n_idx)
                        self.data[new_n_idx].id = self.clId[-1]
        return True
            
        
    def get_neighbours(self, idx):
        neighbours = []
        for this_idx in range(len(self.data)):
            if (not (this_idx == idx)):
                if (euclid(self.data[this_idx].pos, self.data[idx].pos) <= self.eps):
                    neighbours.append(this_idx)
        return neighbours

    def num_clusters(self):
        return len(self.clId) - 1
        


dat = gen_clusters([[1, 1], [7, 7], [15, 15], [30, 30], [40, 40], [60, 60], [67, 65], [75, 75], [81, 81]], 100)
data = make_pts(dat)


param_est = K_nearest(data, 4)
f, ax = plt.subplots()
ax.plot(param_est.nearest_k)
plt.show()

print ("Program went on...")

eps = float(input("Press enter epsilon value that graph implies: "))


instance = DBSCAN(4, eps, data)

instance.main_loop()
print ("Number of clusters: ", instance.num_clusters())
