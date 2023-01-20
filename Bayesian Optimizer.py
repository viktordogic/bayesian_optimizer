import benchmark_functions as bf
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv
from scipy.stats import norm
from scipy.optimize import minimize
import random
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def kernel(x1, x2):
    alpha_zero = 1
    ni = 1.5
    x1[x1 == 0] = 1e-9
    pomoc = np.sqrt(2 * ni) * cdist(x1, x2, 'sqeuclidean')
    pomoc[pomoc == 0] = 1e-9
    return alpha_zero * np.exp(-cdist(x1, x2, 'sqeuclidean'))  # GAUSSIAN KERNEL
    #return alpha_zero * (2 ** (1 - ni) / gamma(ni)) * (pomoc ** ni) * kv(ni, pomoc)         # MATERN KERNEL


def posterior_mean_and_covariance(old_points, old_values, added_point):
    added_point = added_point.reshape(1, -1)
    covariances_1k_1k = kernel(old_points, old_points)
    covariances_1k_1k_inv = np.linalg.inv(covariances_1k_1k)
    covariances_1k_x = kernel(old_points, added_point)
    covariances_x_x = kernel(added_point, added_point)

    main_matrix = covariances_1k_x.T @ covariances_1k_1k_inv

    mean = main_matrix @ old_values
    covariance = covariances_x_x - main_matrix @ covariances_1k_x

    return mean, covariance


def exprected_improvement(old_points, old_values, added_point):
    mean, cov = posterior_mean_and_covariance(old_points, old_values, added_point)
    minimum_value = np.min(old_values)
    delta = mean - minimum_value
    addition_element = 0
    omega = 0.1
    if delta > 0:
        addition_element = delta

    return addition_element + cov * (1 - omega) * norm.pdf(delta / cov + 1e-9) + np.abs(delta) * omega * norm.cdf(delta / cov + 1e-9)


def probability_of_improvement(old_points, old_values, added_point):
    mean, cov = posterior_mean_and_covariance(old_points, old_values, added_point)
    minimum_value = np.min(old_values)
    delta = mean - minimum_value

    return norm.cdf(delta / cov + 1e-9)


def format_boundaries(boundaries):
    reform = []
    for index, each in enumerate(boundaries[0]):
        bla = (each, boundaries[1][index])
        reform.append(bla)
    return reform


class Model:
    def __init__(self, nr_of_initial_points):
        self.function = bf.Rana(n_dimensions=5)
        self.bounds = self.function.suggested_bounds()
        self.dimensions = self.function.n_dimensions()
        self.points = self.space_initialization(nr_of_initial_points)
        self.values = self.get_point_values(self.points)

    def space_initialization(self, x):
        points = np.zeros((x, self.dimensions))
        for i in range(x):
            for y in range(self.dimensions):
                points[i][y] = random.uniform(self.bounds[0][y], self.bounds[1][y])
        return points

    '''def give_random_point(self):
        point = [0 for _ in range(self.dimensions)]
        for y in range(int(self.dimensions)):
            point[y] = random.uniform(self.bounds[0][y], self.bounds[1][y])
        return point'''

    def get_point_values(self, points):
        values = np.zeros(len(points))
        for index, each in enumerate(points):
            values[index] = self.function(each)
        return values

    def get_current_minimum(self):
        return np.min(self.values)

    def get_global_minimum(self):
        lista = [0]
        for ss in self.function.minimum():
            lista.append(ss)
        return self.function(lista[1:])

    def acquisition_function(self, point):
        #return exprected_improvement(self.points, self.values, point)
        return probability_of_improvement(self.points, self.values, point)


if __name__ == "__main__":
    func = Model(5)
    n = 20  # how many iterations
    m = 100  # how many random points to start from
    plotting_values_y = []
    plotting_values_x = []

    start = time.time()
    PERIOD_OF_TIME = 30
    timeout = False
    for _ in range(n):
        for _ in range(10):
            if not abs(np.min(func.values) - func.get_global_minimum()) < 1e-5:
                func.points = func.points.reshape(-1, func.dimensions)
                new_point = func.space_initialization(1)[0]
                min_so_far = func.acquisition_function(new_point)
                if time.time() < start + PERIOD_OF_TIME:
                    for y in range(m):
                        current_point = func.space_initialization(1)[0]
                        if func.acquisition_function(current_point) < min_so_far:
                            min_so_far = func.acquisition_function(current_point)
                            new_point = current_point
                else:
                    timeout = True
                if timeout:
                    break
                new_point = new_point.reshape(1, -1)
                func.points = func.points.reshape(-1, func.dimensions)

                result = minimize(func.acquisition_function, x0=np.squeeze(new_point),
                                  bounds=format_boundaries(func.bounds), method="L-BFGS-B")

                func.points = np.append(func.points, result.x)
                func.values = np.append(func.values, func.function(result.x))

                plotting_values_y.append(np.min(func.values))
                plotting_values_x.append(len(func.points))

        if timeout:
            break
        print(np.min(func.values))
        print("Number of points", len(func.points))

    plt.plot(plotting_values_x, plotting_values_y)
    plt.title(f"5D Rana - Gaussian Kernel, POI\nFinal result is {np.min(func.values):.6f}, global minimum is -2046.830911") #\u03C9
    plt.xlabel("Number of points generated")
    plt.ylabel("Current global minimum")
    plt.show()
