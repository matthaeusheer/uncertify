from dataclasses import dataclass
import random
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

from uncertify.tutorials.datastructures import Point

from typing import List, Tuple


@dataclass
class Cluster:
    center: Point
    points: List[Point]


def k_means(points: List[Point], num_clusters: int,
            center_eps: float = 0.1, max_iterations: int = None) -> List[Cluster]:
    """K-Means clustering algorithm following Lloyds-Algorithm.
    Args:
        points: the raw data points as a list on which to perform k-means
        num_clusters: parameter k, number of clusters
        center_eps: if all clusters centers move less than this value on update step, terminate
        max_iterations: terminate if set and number of iterations reached
    """
    # Initialization: choose k random data points from dataset as cluster centers
    centers = random.sample(points, num_clusters)
    clusters = [Cluster(center=center, points=[]) for center in centers]
    termination_criteria_fulfilled = False
    iteration_idx = 0
    while not termination_criteria_fulfilled:
        # Assignment step: assign each data point to nearest cluster center
        clusters = list(map(reset_points, clusters))
        clusters = assign_points_to_cluster(points, clusters)
        # Update centers such that it forms the centroid of all the points assigned to it
        old_centers = [cluster.center for cluster in clusters]
        clusters = list(map(update_cluster_center, clusters))
        plot_state(clusters)
        iteration_idx += 1
        center_movements = [distance(cluster.center, old_center) for cluster, old_center in zip(clusters, old_centers)]
        if all([movement < center_eps for movement in center_movements]):
            termination_criteria_fulfilled = True
        if max_iterations is not None:
            if iteration_idx == max_iterations:
                termination_criteria_fulfilled = True
    return clusters


def distance(point1: Point, point2: Point) -> float:
    return math.sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2)


def reset_points(cluster: Cluster) -> Cluster:
    cluster.points = []
    return cluster


def calculate_centroid(points: List[Point]) -> Point:
    x_centroid = sum([point.x for point in points]) / len(points)
    y_centroid = sum([point.y for point in points]) / len(points)
    return Point(x_centroid, y_centroid)


def update_cluster_center(cluster: Cluster) -> Cluster:
    cluster.center = calculate_centroid(cluster.points)
    return cluster


def assign_points_to_cluster(points: List[Point], clusters: List[Cluster]) -> List[Cluster]:
    for point in points:
        smallest_cluster_idx = None
        smallest_distance = float('inf')
        for cluster_idx, cluster in enumerate(clusters):
            dist_to_center = distance(point, cluster.center)
            if dist_to_center < smallest_distance:
                smallest_distance = dist_to_center
                smallest_cluster_idx = cluster_idx
        clusters[smallest_cluster_idx].points.append(point)
    return clusters


def plot_state(clusters: List[Cluster]) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()
    colors = iter(cm.viridis(np.linspace(0, 1, len(clusters))))
    for cluster in clusters:
        color = next(colors)
        ax.plot(cluster.center.x, cluster.center.y, 'D', c=color)
        ax.plot([point.x for point in cluster.points], [point.y for point in cluster.points], '.', c=color)
        ax.set_xlim([-50, 50])
        ax.set_ylim([-50, 50])
    ax.set_aspect('equal')
    plt.show()
    return fig, ax


if __name__ == '__main__':
    num_points = 10
    data_points = [Point(random.gauss(mu=0.0, sigma=1.0), random.gauss(mu=0.0, sigma=1.0)) for _ in range(num_points)]
    k_means(data_points, num_clusters=3)
