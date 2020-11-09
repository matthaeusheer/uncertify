import matplotlib.pyplot as plt

from uncertify.visualization.plotting import setup_plt_figure

from typing import Iterable

def plot_fpr_vs_residual_threshold(accepted_fpr: float, calculated_threshold: float,
                                   thresholds: Iterable, fpr_train: list, fpr_val: list = None) -> plt.Figure:
    """Plots the training (possibly also validation) set false positive rates vs. associated residual thresholds.

    Arguments:
        accepted_fpr: accepted false positive rate when testing with the training set itself
        calculated_threshold: the threshold which has been calculated based on this accepted false positive rate
        thresholds: a list of residual pixel value thresholds
        fpr_train: the associated false positive rates on the training set
        fpr_val: same but for validation set, not mandatory
    """
    fig, ax = setup_plt_figure(figsize=(16, 8))

    ax.plot(thresholds, fpr_train, linewidth=4, linestyle='solid', alpha=0.5, label='Training Set')
    if fpr_val is not None:
        ax.plot(thresholds, fpr_val, linewidth=4, linestyle='dashed', alpha=0.5, label='Validation Set')

    ax.set_ylabel(f'False Positive Rate')
    ax.set_xlabel(f'Residual Pixel Threshold')

    normed_diff = [abs(fpr - accepted_fpr) for fpr in fpr_train]
    ax.plot(thresholds, normed_diff, c='green', alpha=0.7, linewidth=3, label='CamCAN FPR - Accepted FPR')
    ax.plot(thresholds, [accepted_fpr] * len(thresholds), linestyle='dotted', linewidth=3, color='grey',
            label=f'Accepted FPR ({accepted_fpr:.2f})')
    ax.plot([calculated_threshold, calculated_threshold], [-0.05, 1], linestyle='dotted', color='green', linewidth=3,
            label=f'Threshold through Golden Section Search ({calculated_threshold:.2f})')
    ax.legend(frameon=False)
    return fig
