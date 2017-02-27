import numpy as np
import matplotlib.pyplot as plt

class LVQ:
	def __init__(self, num_clusters, input_dim, learn_rate):
		self.num_clusters = num_clusters;
		self.input_dim = input_dim;
		self.init_weights();
		self.curr_example = None;
		self.learn_rate = learn_rate;

	def init_weights(self):
		self.weights = np.random.rand(self.num_clusters, self.input_dim);

	def get_diff(self, out_idx_row):
		return sum((self.weights[out_idx_row] - self.curr_example)**2);

	def get_min_output_idx(self):
		diff_list = list(map(self.get_diff, list(range(self.num_clusters))));
		return diff_list.index(min(diff_list));

	def train_example(self):
		min_idx = self.get_min_output_idx();

		self.weights[min_idx] += self.learn_rate*(self.curr_example - \
						self.weights[min_idx]);

	def epoch(self, examples):
		for eg in examples:
			self.curr_example = eg;
			self.train_example();

	def train(self, examples, iterations):
		for i in range(iterations):
			self.epoch(examples);

obj = LVQ(2, 2, 1e-2);

cluster1, cluster2, cluster3 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], 200), \
			np.random.multivariate_normal([-1, -2], [[1, 0], [0, 1]], 200), \
			np.random.multivariate_normal([5, 4], [[1, 0], [0, 1]], 200);

examples = np.concatenate((cluster1, cluster2, cluster3));

obj.train(examples, 50);
print (obj.weights);
