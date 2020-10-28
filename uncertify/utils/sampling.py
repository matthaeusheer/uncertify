import numpy as np
from decimal import Decimal


def random_uniform_ring(center: np.ndarray, outer_radius: float = 1.0, inner_radius: float = 0,
                        n_samples: int = 1) -> np.ndarray:
    """Generate point uniformly distributed in a ring.

    center is a 1D vector whose entries are the coordinates of the center point in the respective dimension

    Taken from
    https://stackoverflow.com/questions/47472123/sample-uniformly-in-a-multidimensional-ring-without-rejection
    """
    n_dims = len(center)
    samples = np.random.normal(size=(n_samples, n_dims))
    samples /= np.linalg.norm(samples, axis=1)[:, np.newaxis]  # push samples to unit sphere
    # Using the inverse cdf method
    u = np.random.uniform(size=n_samples)
    # This is inverse the cdf of ring volume as a function of radius
    sc = (u * (outer_radius ** n_dims - inner_radius ** n_dims) + inner_radius ** n_dims) ** (1 / n_dims)
    samples = samples * sc[:, None] + center
    return samples.astype(np.float)
