# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 21:25:24 2015

@author: Konrad
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc_p

def gen_clusters(means, num_each):
    tup = ();
    for m in means:
        tup = tup + (np.random.multivariate_normal(m, np.diag(np.ones(2)), num_each),)
    data = np.concatenate(tup);
    np.random.shuffle(data);
    return data;

def make_pts(data):
    pts = [];
    for pos in data:
        pts.append(Point(pos));
    return pts;

def euclid(obj1, obj2):
    if (isinstance(obj1, Point) and isinstance(obj2, Point)):
        return np.sqrt(sum(( obj1.pos - obj2.pos )**2));
    elif (isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray)):
        return np.sqrt(sum(( obj1 - obj2 )**2))
    else:
        return None;

class Point:
    def __init__(self, pos):
        self.pos = copy.deepcopy(pos);
        self.processed = False;
        self.core_dist = None;
        self.reach_dist = None;
        self.in_seed = False;
        

class OPTICS:
    def __init__(self, min_pts, data, max_eps = None):
        self.max_eps = max_eps;
        self.min_pts = min_pts;
        self.data = copy.deepcopy(data);
        self.dim = self.data[0].pos.size;
        self.main_list = [];
        if (self.max_eps == None):
            self.get_max_eps();
        self.main_loop();
    
    def __call__(self, main_idx):
        return self.data[self.main_list[main_idx]].reach_dist;
    
    def main_loop(self):
        for idx, obj in enumerate(self.data):
            if (not obj.processed):
                self.expand_point(idx);
        
        for idx, obj in enumerate(self.data):
            if (not obj.processed):
                self.append_main(idx);
    
    
    def get_max_eps(self):
        extr_x = self.get_extr_x();
        extr_y = self.get_extr_y();
        area = (extr_x[1] - extr_x[0])*(extr_y[1] - extr_y[0]);
        self.max_eps = ((area*self.min_pts*sc_p.gamma(2))/(len(self.data)*np.sqrt(np.pi**2)))**0.5
    
    def get_extr_x(self):
        min_x = float("inf");
        max_x = -float("inf");
        for obj in self.data:
            if obj.pos[0] < min_x:
                min_x = obj.pos[0];
            if obj.pos[0] > max_x:
                max_x = obj.pos[0];
        return (min_x, max_x);
        
    def get_extr_y(self):
        min_y = float("inf");
        max_y = -float("inf");
        for obj in self.data:
            if obj.pos[1] < min_y:
                min_y = obj.pos[1];
            if obj.pos[1] > max_y:
                max_y = obj.pos[1];
        return (min_y, max_y);
    
    def append_main(self, idx):
        self.data[idx].processed = True;
        if (self.data[idx].reach_dist == None):
            self.data[idx].reach_dist = self.max_eps;
        self.main_list.append(idx);
    
    def expand_point(self, idx):
        self.get_neighbours(idx);
        self.get_core_dist(idx);
        if (self.data[idx].core_dist == -1):
            return;
        else:
            self.data[idx].processed = True;
            self.append_main(idx);
            
            seed_list = [];
            self.append_seed(seed_list, self.data[idx].neighbours, idx)
            while (len(seed_list) > 0):
                curr_idx = seed_list[0];
                self.get_neighbours(curr_idx);
                self.get_core_dist(curr_idx);
                self.data[curr_idx].processed = True;
                self.append_main(curr_idx);
                self.remove_seed(seed_list);
                if (not (self.data[curr_idx].core_dist == -1)):
                    self.append_seed(seed_list, self.data[curr_idx].neighbours, curr_idx);
                    
                
    def get_core_dist(self, idx):
        if (len(self.data[idx].neighbours) >= self.min_pts):
            self.data[idx].core_dist = self.data[idx].neighbours[self.min_pts - 1][1];
        else:
            self.data[idx].core_dist = -1;
        
    def get_reach_dist(self, center_idx, idx, dist):
        r_dist = max(dist, self.data[center_idx].core_dist);
        if (self.data[idx].reach_dist == None):
            self.data[idx].reach_dist = r_dist;
            return True;
        elif (self.data[idx].reach_dist > r_dist):
            self.data[idx].reach_dist = r_dist;
            return True;
        else:
            return False;
        
    def get_neighbours(self, idx):
        self.data[idx].neighbours = [];
        
        for n_idx, obj in enumerate(self.data):
            dist = euclid(obj, self.data[idx])
            if (dist <= self.max_eps):
                self.data[idx].neighbours.append([n_idx, dist]);
        
        self.data[idx].neighbours.sort(key = lambda x : x[1]);
    
    def append_seed(self, seed_list, neighbours, center_idx):
        for n_tup in neighbours:
            changed = self.get_reach_dist(center_idx, n_tup[0], n_tup[1]);
            
            if (self.data[n_tup[0]].in_seed and changed):
                del seed_list[seed_list.index(n_tup[0])];
                self.data[n_tup[0]].in_seed = False;
            elif (self.data[n_tup[0]].processed or self.data[n_tup[0]].in_seed):
                continue;
            
            for idx, obj in enumerate(seed_list):
                if ( self.data[n_tup[0]].reach_dist < self.data[obj].reach_dist ):
                    seed_list.insert(idx, n_tup[0]);
                    self.data[n_tup[0]].in_seed = True;
                    break;       
            if (not self.data[n_tup[0]].in_seed):
                seed_list.append(n_tup[0]);
                self.data[n_tup[0]].in_seed = True;
    
    def remove_seed(self, seed_list):
        self.data[seed_list[0]].in_seed = False;
        del seed_list[0];
        
    def reach_plot(self):
        x = list(range(len(self.main_list)));
        y = [];
        
        for idx in self.main_list:
            y.append(self.data[idx].reach_dist);
            
        f, ax = plt.subplots();
        ax.bar(x, y);
        
    def print_reach_dist(self):
        for idx in self.main_list:
            print (idx)
            print (self.data[idx].reach_dist)
            
    def plot_data(self):
        x = [];
        y = [];
        
        for obj in self.data:
            x.append(obj.pos[0]);
            y.append(obj.pos[1]);
            
        f, ax = plt.subplots();
        ax.scatter(x, y);
        
    def get_num_clusters(self):
        clusters = [];
        up = True;
        top, bottom = -1, -1;
        for i, idx in enumerate(self.main_list[:-1]):
            if (up and (self.data[idx].reach_dist > self.data[self.main_list[i + 1]])):
                up = not up;
                if (not bottom == -1):
                    clusters.append(top - bottom);
                top = self.data[idx].reach_dist;
                continue;
            if (not up) and (self.data[idx].reach_dist < self.data[self.main_list[i + 1]].reach_dist):
                up = not up;
                bottom = self.data[idx].reach_dist;
            

class Clusters:
    def __init__(optics_obj, eps):
        self.optics_obj = optics_obj;
        self.main_list =  optics_obj.main_list;
        self.eps = eps;
        self.min_pts = optics_obj.min_pts;
        
    def find(self):
        idx = 0;
        #down, up = False, False;
        downs = [];
        clusters = [];
        while idx < len(self.main_list):
            diff = self.main_list[idx] - self.main_list[idx + 1];
            if (diff >= self.optics_obj(idx)*self.eps):
                new_down, idx = self.proc_down(idx);
                downs.append([new_down, -float("inf")]);
                #glob_mib = self.optics_obj(downs[-1][0][0]]);
                #self.filter_downs(glob_mib, downs);
            elif (-diff >= self.optics_obj(idx)*self.eps):
                glob_mib = self.get_glob_mib(downs[-1], idx);
                self.filter_downs(glob_mib, downs);
                up, idx = self.proc_up(idx);
                for down in downs:
                    if (self.optics_obj(up[1]).reach_dist*(1 - self.eps) >= down[1]):
                        clusters.append((down[0][0], up[1]));
            else:
                idx += 1;
                        
    def get_glob_mib(self, last_down, curr_idx):
        begin_idx, end_idx = last_down[0][1], curr_idx;
        glob_mib = -float("inf");
        
        for i in range(begin_idx, end_idx + 1):
            if (self.optics_obj(i) > glob_mib):
                glob_mib = self.optics_obj(i);
        
        return glob_mib;
                
    def proc_down(self, idx):
        bad_inrow = 0;
        begin_idx = idx;
        while (idx < len(self.main_list)):
            idx += 1;
            diff = self.main_list[idx].reach_dist - self.main_list[idx + 1].reach_dist;
            if (diff < 0):
                return (begin_idx, idx - 1);
            if (diff > 0):
                if (diff >= self.eps*self.main_list[idx]):
                    bad_inrow = 0;
                else:
                    if (bad_inrow == 0):
                        last_good = idx - 1;
                    bad_inrow += 1;
                    if bad_inrow > self.min_pts:
                        # include a check that ensures region does not have
                        # length zero?
                        return (begin_idx, last_good), idx;
    
    def proc_up(self, idx):
        bad_inrow = 0;
        begin_idx = idx;
        while (idx < len(self.main_list)):
            idx += 1;
            diff = self.main_list[idx].reach_dist[idx + 1] - self.main_list[idx].reach_dist;
            if (diff < 0):
                return (begin_idx, idx - 1);
            if (diff > 0):
                if (diff >= self.eps*self.main_list[idx + 1]):
                    bad_inrow = 0;
                else:
                    if (bad_inrow == 0):
                        last_good = idx - 1;
                    bad_inrow += 1;
                    if (bad_inrow > self.min_pts):
                        return (begin_idx, last_good), idx;
                        
    def filter_downs(self, glob_mib, downs):
        del_idx = [];
        for idx, obj in enumerate(downs[:-1]):
            if self.main_list[obj[0][0]].reach_dist*(1 - self.eps) < glob_mib:
                del_idx.append(idx);
            elif (obj[1] < glob_mib):
                downs[idx][1] = glob_mib;
        del_idx.reverse();
        for i in del_idx:
            del downs[i];

dat = gen_clusters([[1, 1], [6, 7], [10, 15], [15, 15]], 200);
data = make_pts(dat);

optics = OPTICS(15, data);

optics.reach_plot();
optics.plot_data();

plt.show();

#optics.print_reach_dist();

print ("Done")