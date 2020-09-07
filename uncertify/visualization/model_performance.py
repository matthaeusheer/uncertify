import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from uncertify.visualization.plotting import setup_plt_figure

from typing import Iterable, Tuple


def plot_segmentation_performance_vs_threshold(thresholds: Iterable[float],
                                               dice_scores: Iterable[float] = None,
                                               iou_scores: Iterable[float] = None,
                                               train_set_threshold: float = None,
                                               **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    if dice_scores is None and iou_scores is None:
        raise ValueError(f'Need to provide either dice scores or iou scores or both.')
    fig, ax = setup_plt_figure(**kwargs)
    max_score = 0
    if dice_scores is not None:
        if max(dice_scores) > max_score:
            max_score = max(dice_scores)
        ax.plot(thresholds, dice_scores, linewidth=3, c='cyan', label='Dice score')
    if iou_scores is not None:
        if max(iou_scores) > max_score:
            max_score = max(iou_scores)
        ax.plot(thresholds, iou_scores, linewidth=3, c='orange', label='IoU score')
    if train_set_threshold is not None:
        ax.plot([train_set_threshold, train_set_threshold], [0, max_score], linewidth=1, linestyle='dashed', c='Grey',
                label=f'Training set threshold (GSS)')
    ax.set_xlabel('Pixel threshold for anomaly detection', fontweight='bold')
    ax.set_ylabel('Segmentation score', fontweight='bold')
    ax.legend(frameon=False)
    return fig, ax


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