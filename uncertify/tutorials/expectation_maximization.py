import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def expectation_maximization(points: np.ndarray, num_clusters: int,
                             center_eps: float = 0.1, max_iterations: int = None):
    """Expectation Maximization for Gaussian Mixture Models.
    Args:
        points: numpy array in shape (2, num_points), first row x coordinates, second row y coordinates
        num_clusters: number of mixture components
        center_eps: terminate when no cluster mean moves more than this in update step
        max_iterations: terminate when reached number of max_iterations
    """
    # Initializations
    _, n_points = points.shape
    dimensions = 2
    soft_assigns = np.zeros((num_clusters, n_points))
    means = sample_initial_centers(points, num_clusters)
    # TODO: Fix covariance matrix initialization
    # covariances = np.random.random((num_clusters, dimensions, dimensions))
    covariances = np.array([
        [[1, 1],
         [1, 2]],
        [[1, 1],
         [1, 2]]
    ])
    cluster_probabilities = np.random.random(num_clusters)
    cluster_probabilities /= cluster_probabilities.sum()

    termination_criteria_fulfilled = False
    iteration_counter = 0
    while not termination_criteria_fulfilled:
        print(f'--- Iteration {iteration_counter} ---')
        # Expectation step: Calculate soft assignment for every point to all clusters
        normal_distributions = get_multivariate_normals(means, covariances)
        for cluster_idx, probability in enumerate(cluster_probabilities):
            for point_idx, point in enumerate(points.T):  # todo fix
                normalization = sum([normal_distribution.pdf(point) for normal_distribution in normal_distributions])
                soft_assignment = probability * normal_distributions[cluster_idx].pdf(point) / normalization
                soft_assigns[cluster_idx, point_idx] = soft_assignment

        # Maximization step: Given soft assignments, calculate new estimates for cluster parameters
        for cluster_idx in range(num_clusters):
            mean_new = calc_new_mean(points, soft_assigns, cluster_idx, num_clusters)
            covariance_new = calc_new_covariance(mean_new, soft_assigns, points, cluster_idx, num_clusters)
            new_probability = calc_n_soft(cluster_idx, soft_assigns) / num_clusters
            means[:, cluster_idx] = mean_new
            covariances[cluster_idx] = covariance_new
            cluster_probabilities[cluster_idx] = new_probability

        iteration_counter += 1
        if max_iterations is not None:
            if iteration_counter == max_iterations:
                termination_criteria_fulfilled = True

        plot_state(points, means, covariances)


def determinant_2d(matrix: np.ndarray) -> float:
    """Calculates the determinant of a 2x2 matrix."""
    assert matrix.shape == (2, 2), f'Determinant function defined for 2D numpy array only.'
    return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]


def get_multivariate_normals(mean_vectors, covariance_matrices) -> list:
    """Initializes a multivariate gaussian for every cluster and returns list of normals sorted like clusters."""
    normal_distributions = []
    for mean, covariance in zip(mean_vectors.T, covariance_matrices):
        normal_distributions.append(multivariate_normal(mean, covariance))
    return normal_distributions


def sample_initial_centers(points: np.ndarray, num_clusters: int) -> np.ndarray:
    """Initialize the means of the clusters with random sampled points from the dataset.

    Returns:
        (dimensions x num_clusters) numpy array, i.e. each column is a (x, y) vector for a cluster center
    """
    _, n_points = points.shape
    sampled_point_indices = np.random.choice(range(n_points), size=num_clusters, replace=False)
    center_points = points[:, sampled_point_indices]
    return center_points


def calc_new_mean(points: np.ndarray, soft_assigns: np.ndarray, cluster_idx: int, num_clusters: int):
    """Calculate the new mean for a cluster in the maximization step."""
    return 1 / calc_n_soft(cluster_idx, soft_assigns) * np.sum(
        [soft_assigns[cluster_idx, point_idx] * points[:, point_idx] for point_idx in range(num_points)], axis=0
    )


def calc_new_covariance(new_mean: np.ndarray, soft_assigns: np.ndarray, points: np.ndarray,
                        cluster_idx: int, num_clusters: int):
    """Calculate the new covariance matrix for a cluster in the maximization step."""
    new_covariance = np.ndarray(shape=(num_clusters, num_clusters))
    for point_idx, point in enumerate(points.T):
        new_covariance += soft_assigns[cluster_idx, point_idx] * np.outer(point - new_mean, point - new_mean)
    return 1 / calc_n_soft(cluster_idx, soft_assigns) * new_covariance


def calc_n_soft(cluster_idx: int, soft_assigns: np.ndarray):
    """Calculate the new soft number of assigned particles in the maximization step for normalization."""
    return soft_assigns.sum(axis=1)[cluster_idx]


def plot_state(points, means, covariances):
    fig, ax = plt.subplots()
    ax.plot(points[0], points[1], '.')

    _, n_points = points.shape
    x = uniform(-20, 20, n_points)
    y = uniform(-20, 20, n_points)
    z = mul(x, y, Sigma=np.asarray([[1., .5], [0.5, 1.]]), mu=np.asarray([0., 0.]))
    plot_countour(x, y, z)

    plt.show()


def plot_countour(x, y, z):
    # define grid.
    xi = np.linspace(-2.1, 2.1, 100)
    yi = np.linspace(-2.1, 2.1, 100)
    # grid the data.
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')
    levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    # contour the gridded data, plotting dots at the randomly spaced data points.
    CS = plt.contour(xi, yi, zi, len(levels), linewidths=0.5, colors='k', levels=levels)
    # CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
    CS = plt.contourf(xi, yi, zi, len(levels), cmap=cm.Greys_r, levels=levels)
    plt.colorbar()  # draw colorbar
    # plot data points.
    # plt.scatter(x, y, marker='o', c='b', s=5)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title('griddata test (%d points)' % npts)
    plt.show()


if __name__ == '__main__':
    num_points = 10
    num_mixtures = 2
    max_iterations = 2
    canvas_size = []


    def create_gaussian_blobs(num_samples: int) -> np.ndarray:
        blob1_x = np.random.normal(loc=-10, scale=2, size=(1, num_samples // 2))
        blob1_y = np.random.normal(loc=-10, scale=1, size=(1, num_samples // 2))
        blob2_x = np.random.normal(loc=10, scale=2, size=(1, num_samples // 2))
        blob2_y = np.random.normal(loc=10, scale=1, size=(1, num_samples // 2))
        blob1 = np.vstack([blob1_x, blob1_y])
        blob2 = np.vstack([blob2_x, blob2_y])
        return np.hstack([blob1, blob2])


    data_points = create_gaussian_blobs(num_points)
    expectation_maximization(data_points, num_clusters=num_mixtures, max_iterations=max_iterations)
