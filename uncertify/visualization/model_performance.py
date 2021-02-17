import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from uncertify.visualization.plotting import setup_plt_figure

from typing import Iterable, List, Union, Tuple

LOG = logging.getLogger(__name__)


def plot_segmentation_performance_vs_threshold(thresholds: Iterable[float],
                                               dice_scores: Iterable[float] = None,
                                               dice_stds: Iterable[float] = None,
                                               iou_scores: Iterable[float] = None,
                                               iou_stds: Iterable[float] = None,
                                               train_set_threshold: float = None,
                                               **kwargs) -> plt.Figure:
    """Plot dice and iou scores vs various residual pixel thresholds."""
    if dice_scores is None and iou_scores is None:
        raise ValueError(f'Need to provide either dice scores or iou scores or both.')
    fig, ax = setup_plt_figure(**kwargs)
    max_score = 0
    if dice_scores is not None:
        if max(dice_scores) > max_score:
            max_score = max(dice_scores)
        ax.errorbar(thresholds, dice_scores, yerr=dice_stds, linewidth=2, c='green', label='Dice score')
    if iou_scores is not None:
        if max(iou_scores) > max_score:
            max_score = max(iou_scores)
        ax.errorbar(thresholds, iou_scores, yerr=iou_stds, linewidth=2, c='green', label='IoU score')
    if train_set_threshold is not None:
        ax.plot([train_set_threshold, train_set_threshold], [0, max_score], linewidth=1, linestyle='dashed', c='Grey',
                label=f'Training set threshold (GSS)')
    ax.set_xlabel('Pixel threshold for anomaly detection', fontweight='bold')
    ax.set_ylabel('Segmentation score', fontweight='bold')
    ax.legend(frameon=False)
    return fig


def setup_roc_prc_fig(mode: str, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Sets up a bare figure for either ROC or PRC curves to plot on curves later on."""
    if mode not in ['roc', 'prc']:
        raise ValueError('Choose either "roc" or "prc" for figure setup!')

    x_labels = {'roc': 'False Positive Rate', 'prc': 'Recall (TPR)'}
    y_labels = {'roc': 'True Positive Rate', 'prc': 'Precision'}

    fig, ax = setup_plt_figure(xlabel=x_labels[mode], ylabel=y_labels[mode], **kwargs)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if mode == 'roc':
        ax.plot([0, 1], [0, 1], linewidth=1, linestyle='--', color='gray')
    return fig, ax


def plot_multi_roc_curves(fprs: List[list], tprs: List[list], aucs: List[float],
                          labels: List[str], **kwargs) -> plt.Figure:
    """Plot multiple ROC curves in one figure."""
    fig, ax = setup_roc_prc_fig(mode='roc', **kwargs)
    for fpr, tpr, auc, label in zip(fprs, tprs, aucs, labels):
        ax.plot(fpr, tpr, linewidth=3, alpha=0.7, label=f'{label:10} {auc:.2f}')
    ax.legend(title='A.u. ROC Curve')
    return fig


def plot_multi_prc_curves(recalls: List[list], precisions: List[list], auprcs: List[float],
                          labels: List[str], **kwargs) -> plt.Figure:
    """Plot multiple PRC curves in one figure."""
    fig, ax = setup_roc_prc_fig(mode='prc', **kwargs)
    for recall, precision, auprc, label in zip(recalls, precisions, auprcs, labels):
        ax.plot(recall, precision, linewidth=3, alpha=0.7, label=f'{label:10} {auprc:.2f}')
    ax.legend(title='A.u. PR Curve')
    return fig


def plot_roc_curve(fpr: Union[list, List[list]], tpr: Union[list, List[list]], auc: float,
                   calculated_threshold: float = None, thresholds: Iterable = None, **kwargs) -> plt.Figure:
    """Plots the ROC curve."""
    fig, ax = setup_plt_figure(xlabel='False Positive Rate', ylabel='True Positive Rate', **kwargs)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='x', pad=6)
    ax.tick_params(axis='y', pad=7)

    ax.plot(fpr, tpr, linewidth=2, alpha=0.7, color='green', label=f'AUC={auc:.2f}')
    ax.plot([0, 1], [0, 1], linewidth=1, linestyle='--', color='gray')

    if calculated_threshold is not None:
        if thresholds is None:
            LOG.warning(f'plot_roc_curve: Cannot plot point on ROC curve for calculated threshold since '
                        f'thresholds list is not given.')
        else:
            idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - calculated_threshold))
            ax.plot([fpr[idx]], [tpr[idx]], 'o', color='cyan')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    leg = ax.legend(handlelength=0, handletextpad=0, frameon=False)
    for item in leg.legendHandles:
        item.set_visible(False)
    return fig


def plot_precision_recall_curve(precision: Iterable, recall: Iterable, auprc: float,
                                calculated_threshold: float = None, thresholds: Iterable = None,
                                **kwargs) -> plt.Figure:
    """Plots the PRC curve."""
    fig, ax = setup_plt_figure(xlabel='Recall', ylabel='Precision', **kwargs)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='x', pad=6)
    ax.tick_params(axis='y', pad=7)

    ax.plot(precision, recall, linewidth=2, alpha=0.7, color='green', label=f'AUPRC={auprc:.2f}')

    if calculated_threshold is not None:
        if thresholds is None:
            LOG.warning(f'plot_precision_recall_curve: Cannot plot pont on ROC curve for calculated threshold since '
                        f'thresholds list is not given.')
        else:
            idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - calculated_threshold))
            ax.plot([precision[idx]], [recall[idx]], 'o', color='cyan')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    leg = ax.legend(handlelength=0, handletextpad=0, frameon=False)
    for item in leg.legendHandles:
        item.set_visible(False)
    return fig


def plot_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    """
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Args:
        cf:            confusion matrix to be passed in
        group_names:   List of strings that represent the labels row by row to be shown in each square.
        categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
        count:         If True, show the raw number in the confusion matrix. Default is True.
        cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                       Default is True.
        xyticks:       If True, show x and y ticks. Default is True.
        xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
        sum_stats:     If True, display summary statistics below the figure. Default is True.
        figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
        cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                       See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    """

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize is None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks is False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('Actual', fontweight='bold')
        plt.xlabel('Predicted' + stats_text, fontweight='bold')
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
    return plt.gcf(), plt.gca()
