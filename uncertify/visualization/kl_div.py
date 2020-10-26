import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from uncertify.visualization.plotting import setup_plt_figure


def plot_gauss_1d_kl_div(mean1, std1, mean2, std2, xmin: int = -10, xmax: int = 10, n_samples: int = 100) -> plt.Figure:
    """Plot two Gauss distributions (mean1, std1) and (mean2, std2) and calculate the KL divergence (figure title)."""
    def kl_divergence(p, q):
        return np.mean(np.where(p != 0, p * np.log(p / q), 0))

    x_values = np.linspace(xmin, xmax, n_samples)
    gauss_values_1 = norm.pdf(x_values, mean1, std1)
    gauss_values_2 = norm.pdf(x_values, mean2, std2)
    fig, ax = setup_plt_figure(title=f'KL(p||q) = {kl_divergence(gauss_values_1, gauss_values_2):1.3f}',
                               xlabel='x', ylabel='pdf(x)')
    ax.plot(x_values, gauss_values_1, 'o-', c='teal', label=f'p {mean1, std1}')
    ax.plot(x_values, gauss_values_2, 'o-', c='orange', label=f'q {mean2, std2}')
    ax.legend()
    return fig
