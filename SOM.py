import matplotlib.pyplot as plt
import numpy as np
import random as rd
import math
import copy

def dist_sq(vec1, vec2):
	return sum((vec1 - vec2)**2);

def neighbour_func(coord1, coord2, scale):
	return math.exp(-sum((coord1 - coord2)**2)*scale);

class SOM:
	def __init__(self, size, patterns, neighbour_fn, learn_rate):
		self.size = size;
		self.patterns = np.asarray([np.asarray(p) for p in patterns]);
		np.random.shuffle(self.patterns);
		self.grid = np.zeros((self.size, self.size, self.patterns[0].size));
		self.init_grid_random_samples();
		self.curr_pattern = None;
		self.neighbour_fn = neighbour_fn;
		self.iterations = 0;
		self.learn_rate = learn_rate;
		
	def init_grid_edges(self):
		"""
		for col_idx in range(len(self.grid[0])):
			self.grid[0][col_idx] = (self.grid[0][-1] - self.grid[0][0])/(len(self.grid[0]) - 1)*col_idx + self.grid[0][0];
			self.grid[-1][col_idx] = (self.grid[-1][-1] - self.grid[-1][0])/(len(self.grid[-1]) - 1)*col_idx + self.grid[-1][0];
		"""
		
		for row_idx in range(len(self.grid)):
			self.grid[row_idx][0] = (self.grid[-1][0] - self.grid[0][0])/(len(self.grid) - 1)*row_idx + self.grid[0][0];
			self.grid[row_idx][-1] = (self.grid[-1][-1] - self.grid[0][-1])/(len(self.grid) - 1)*row_idx + self.grid[0][-1];
		

	def init_grid(self):
		extremes = self.find_first_extremes();
		for i in range(2):
			extremes.append(self.find_next_extreme(extremes));
		self.grid[0][0] = self.patterns[extremes[0]];
		self.grid[-1][0] = self.patterns[extremes[1]];
		self.grid[0][-1] = self.patterns[extremes[2]];
		self.grid[-1][-1] = self.patterns[extremes[3]];

		self.init_grid_edges();
		
		for row_idx in range(len(self.grid)):
			for col_idx in range(len(self.grid[row_idx])):
				self.grid[row_idx][col_idx] = (self.grid[row_idx][-1] - self.grid[row_idx][0])/(len(self.grid[row_idx]) - 1)*col_idx + self.grid[row_idx][0];
		
	def init_grid_simple(self):
		self.grid = np.random.rand(self.size, self.size, 2);

	def init_grid_random_samples(self):
		rand_samples = rd.sample(list(self.patterns), self.size*self.size);
		for i in range(self.size):
			for j in range(self.size):
				self.grid[i][j] = rand_samples[i*self.size + j] + rd.random() - 0.5;

	def find_first_extremes(self):
		biggest_dist = 0;
		biggest_pair = None;
		for p_idx1 in range(len(self.patterns)):
			for p_idx2 in range(p_idx1 + 1, len(self.patterns)):
				dist = sum((self.patterns[p_idx1] \
					- self.patterns[p_idx2])**2);
				if dist > biggest_dist:
					biggest_dist = dist;
					biggest_pair = [p_idx1, p_idx2];
		return biggest_pair;

	def find_next_extreme(self, extremes_idx):
		biggest_dist = 0;
		biggest_pattern_idx = None;
		for p_idx in range(len(self.patterns)):
			temp_distances = [];
			for i in extremes_idx:
				temp_distances.append(sum((self.patterns[p_idx] - self.patterns[i])**2)); 
			dist = np.prod(temp_distances);
			if (dist > biggest_dist):
				biggest_dist = dist;
				biggest_pattern_idx = p_idx;
		return p_idx;

	def get_winner(self):
		shortest_dist = float("inf");
		self.winner_coord = None;
		
		for row_idx in range(len(self.grid)):
			for col_idx in range(len(self.grid[0])):
				candidate_dist = dist_sq(self.grid[row_idx][col_idx], self.curr_pattern);
				if candidate_dist < shortest_dist:
					shortest_dist = candidate_dist;
					self.winner_coord = np.asarray([row_idx, col_idx]);


	def update_weights(self):
		for row_idx in range(len(self.grid)):
			for col_idx in range(len(self.grid[0])):
				self.grid[row_idx][col_idx] += self.learn_rate*self.neighbour_fn(np.asarray([row_idx, col_idx]), self.winner_coord,  \
									float(self.iterations)*0.00005)*(self.curr_pattern - self.grid[row_idx][col_idx]);

	def run(self, max_iters):
		iters = 0;
		for i in range(max_iters*len(self.patterns)):
			if (i % len(self.patterns) == 0):
				print ("in iteration: ", i/len(self.patterns))
			self.curr_pattern = self.patterns[self.iterations % len(self.patterns)];
			self.get_winner();
			self.iterations += 1;
			self.update_weights();
		self.make_u_matrix();

	def add_val(self, avg_var, count, coord):
		if (coord[0] > 0 and coord[0] < self.size*2 - 1 \
			and coord[1] > 0 and coord[1] < self.size*2 - 1):
			avg_var += self.u_matrix[coord[0]][coord[1]];
			count += 1;
		return (avg_var, count);

	def avg_cell(self, cell_c):
		avg = 0;
		count = 0;
		for i in range(-1, 2, 1):
			avg, count = self.add_val(avg, count, (cell_c[0] - 1, cell_c[1] + i));
			avg, count = self.add_val(avg, count, (cell_c[0] + 1, cell_c[1] + i));
		
		avg, count = self.add_val(avg, count, (cell_c[0], cell_c[1] - 1));
		avg, count = self.add_val(avg, count, (cell_c[0], cell_c[1] + 1));

		self.u_matrix[cell_c[0]][cell_c[1]] = float(avg)/count;
				

	def make_u_matrix(self):
		self.u_matrix = np.zeros((self.size*2 - 1, self.size*2 - 1));

		for i in range(0, self.size*2 - 1, 2):
			for j in range(1, self.size*2 - 1, 2):
				#print("dist_sq: ", dist_sq(self.grid[(i - 1)/2][(j - 1)/2], self.grid[(i - 1)/2][(j + 1)/2]));
				self.u_matrix[i][j] = dist_sq(self.grid[(i)/2][(j - 1)/2], self.grid[(i)/2][(j + 1)/2])**(0.5);
		
		for i in range(1, self.size*2 - 1, 2):
			for j in range(0, self.size*2 - 1, 2):
				self.u_matrix[i][j] = dist_sq(self.grid[(i - 1)/2][j/2], self.grid[(i + 1)/2][j/2])**0.5;

		for i in range(1, self.size*2 - 1, 2):
			for j in range(1, self.size*2 - 1, 2):
				self.u_matrix[i][j] = (dist_sq(self.grid[(i - 1)/2][(j - 1)/2], self.grid[(i + 1)/2][(j + 1)/2])**0.5 \
						+ dist_sq(self.grid[(i + 1)/2][(j - 1)/2], self.grid[(i - 1)/2][(j + 1)/2])**0.5) \
						/(8**0.5)

		for i in range(0, self.size*2 - 1, 2):
			for j in range(0, self.size*2 - 1, 2):
				self.avg_cell([i, j]);

		
	
	def visualize(self):
		x_vals = [range(self.size*2 - 1) for i in range(self.size*2 - 1)];
		y_vals = [np.ones(self.size*2 - 1)*i for i in range(self.size*2 - 1)];
		
		cols = np.ravel(self.u_matrix);
		cols = cols/max(cols);
	
		print ("color codes: ", cols);

		plt.scatter(x_vals, y_vals, facecolors=[(1-col, 1-col, 1-col) for col in cols]);
		plt.show();

patterns1, patterns2, patterns3 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], 100), \
			np.random.multivariate_normal([-1, -1], [[1, 0], [0, 1]], 100), \
			np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], 100);

patterns = np.concatenate((patterns1, patterns2, patterns3));
np.random.shuffle(patterns);

som = SOM(10, patterns, neighbour_func, 0.1);
print ("Grid before running: ");
print (som.grid);
som.run(100);
print ("Grid after running: ");
print (som.grid);
print (som.u_matrix);
som.visualize();
