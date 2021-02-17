import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from uncertify.common import DATA_DIR_PATH

from typing import List

LABEL_MAP = {
    'rec_err': '$\ell_{1}$',
    'kl_div': '$D_{KL}$',
    'elbo': '$\mathcal{L}$',
    'entropy': '$H_{\ell_{1}}$',
    'entropy_rec_err_kl_div_elbo': '$H_{\ell_{1}}, \ell_{1}, D_{KL}, \mathcal{L}'
}

LOG = logging.getLogger(__name__)


def do_pair_plot_statistics(statistics_dict: dict, dose_statistics: List[str],
                            dataset_name: str, hue: str = 'is_lesional') -> sns.PairGrid:
    """
    Arguments
    ---------
        statistics_dict: dictionary as returned by aggregate_slice_wise_statistics
        dose_statistics: statistics to use in the plot
        dataset_name: name of the dataset used for file naming
        hue: which column in the dataframe to use as hue
    """
    stat_df = pd.DataFrame(statistics_dict)
    grid = sns.pairplot(stat_df, vars=dose_statistics, corner=True, plot_kws={"s": 10}, palette='viridis',
                        hue=hue, diag_kws={'shade': False}, diag_kind='kde')
    grid.map_lower(sns.kdeplot, shade=True, thresh=0.05, alpha=0.7)

    if hue is not None:
        grid._legend.set_title('')
        new_labels = ['healthy', 'lesional']
        for t, l in zip(grid._legend.texts, new_labels):
            t.set_text(l)
    # Set nice x and y labels
    for ax in grid.axes.flatten():
        if ax is not None:
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            old_xlabel = ax.get_xlabel()
            old_ylabel = ax.get_ylabel()
            if old_xlabel in LABEL_MAP:
                ax.set_xlabel(LABEL_MAP[old_xlabel])
            if old_ylabel in LABEL_MAP:
                ax.set_ylabel(LABEL_MAP[old_ylabel])
    grid.tight_layout()
    grid.savefig(DATA_DIR_PATH / 'plots' / f'dose_pairplot_{dataset_name}.png')
    return grid


