import logging
import numpy as np

from uncertify.evaluation.model_performance import calculate_roc, calculate_prc
from uncertify.visualization.model_performance import plot_roc_curve, plot_precision_recall_curve
from uncertify.visualization.histograms import plot_multi_histogram

LOG = logging.getLogger(__name__)


def slice_wise_lesion_detection_dose_kde(dataloader_dict: dict, train_loader_name: str,
                                         metrics_ood_dict: dict, dose_statistics: list, predict_mode: str,
                                         do_plots: bool = True) -> None:
    """For a given metrics_ood_dict, check if the dose_kde metrics can be used for slice-wise anomaly detection.

    Arguments:
        dataloader_dict: a (name, dataloader) key, value dictionary
        train_loader_name: name of the in-distribution dataloader present in the metrics_ood_dict and dataloader_dict
        metrics_ood_dict: a dictionary es returned by run_ood_to_ood_dict
        dose_statistics: a list of dose statistics we take into account here, e.g. entropy, kl_div, rec_err, elbo
        do_plots: if yes, will to roc, prc and histogram plot
        predict_mode: either 'kde' or 'stat', if kde the dose_kde values will be taken if 'stat' the raw statistics
    """
    for dataloader_name in dataloader_dict.keys():
        if 'can' in dataloader_name.lower():
            continue  # no need to do slice-wise lesion detection here
        print()
        print(f' --- {dataloader_name} --- ')
        # post_kde_stat are statistics names we use for the dose_kde
        joined_stats_name = '_'.join(dose_statistics)
        for kde_stat_name in list(dose_statistics) + [joined_stats_name]:
            kde_in_dist = metrics_ood_dict['dose'][train_loader_name]['dose_kde_healthy'][kde_stat_name]
            stat_in_dist = metrics_ood_dict['dose'][train_loader_name]['dose_stat_healthy'][kde_stat_name]
            print(f'{kde_stat_name}')
            stat_healthy_ood, stat_lesional_ood = None, None
            # all dose statistics combined
            if kde_stat_name == joined_stats_name:
                kde_healthy_ood = metrics_ood_dict['dose'][dataloader_name]['healthy']
                kde_lesional_ood = metrics_ood_dict['dose'][dataloader_name]['lesional']
                kde_in_dist = metrics_ood_dict['dose'][train_loader_name]['healthy']
                stat_in_dist = None
            # individual statistics
            else:
                kde_healthy_ood = metrics_ood_dict['dose'][dataloader_name]['dose_kde_healthy'][kde_stat_name]
                kde_lesional_ood = metrics_ood_dict['dose'][dataloader_name]['dose_kde_lesional'][kde_stat_name]
                stat_healthy_ood = metrics_ood_dict['dose'][dataloader_name]['dose_stat_healthy'][kde_stat_name]
                stat_lesional_ood = metrics_ood_dict['dose'][dataloader_name]['dose_stat_lesional'][kde_stat_name]
                # print(f'\t{len(kde_healthy_ood)} healthy & {len(kde_lesional_ood)} lesional samples')

            n_healthy_ood = len(kde_healthy_ood)
            n_lesional_ood = len(kde_lesional_ood)
            n_healthy_id = n_healthy_ood // 2

            y_true = n_healthy_id * [0.0] + n_healthy_ood * [0.0] + n_lesional_ood * [1.0]

            kde_in_dist_samples = list(np.random.choice(kde_in_dist, size=n_healthy_id, replace=False))
            if predict_mode == 'kde':
                y_pred_proba = kde_in_dist_samples + kde_healthy_ood + kde_lesional_ood
            elif predict_mode == 'stat':
                try:
                    stat_in_dist = list(np.random.choice(stat_in_dist, size=n_healthy_id, replace=False))
                except ValueError:
                    LOG.warning(f'Ignored {kde_stat_name}.')
                    continue
                y_pred_proba = stat_in_dist + stat_healthy_ood + stat_lesional_ood
            else:
                raise ValueError('Predict mode not supported.')

            fpr, tpr, _, au_roc = calculate_roc(y_true, np.array(y_pred_proba))
            precision, recall, _, au_prc = calculate_prc(y_true, np.array(y_pred_proba))

            if do_plots:
                plot_roc_prc_slice_wise_lesion_detection_dose_kde(
                    dataloader_name, kde_stat_name,
                    fpr, tpr, au_roc,
                    recall, precision, au_prc,
                    kde_in_dist_samples, kde_healthy_ood, kde_lesional_ood,
                    stat_in_dist, stat_healthy_ood, stat_lesional_ood
                )


def plot_roc_prc_slice_wise_lesion_detection_dose_kde(dataloader_name, post_kde_stat, fpr, tpr, au_roc, recall,
                                                      precision, au_prc,
                                                      healthy_id_kde, healthy_post_kde, lesional_post_kde,
                                                      id_healthy_stat=None, healthy_stat=None, lesional_stat=None):
    plot_multi_histogram([np.array(healthy_id_kde), np.array(healthy_post_kde), np.array(lesional_post_kde)],
                         ['train', 'healthy OOD', 'lesional OOD'],
                         plot_density=False,
                         show_data_ticks=False,
                         legend_title=f'{dataloader_name} ' + '$DoSE_{KDE}$' + f'({post_kde_stat})',
                         legend_pos='upper left',
                         figsize=(11, 4))
    if healthy_stat is not None and lesional_stat is not None and id_healthy_stat is not None:
        plot_multi_histogram([np.array(id_healthy_stat), np.array(healthy_stat), np.array(lesional_stat)],
                             ['train', 'healthy OOD', 'lesional OOD'],
                             plot_density=False,
                             show_data_ticks=False,
                             legend_title=f'{dataloader_name} ' + 'Stat' + f'({post_kde_stat})',
                             legend_pos='upper left',
                             figsize=(11, 4))

    plot_roc_curve(fpr, tpr, au_roc, title=f'ROC SW Anomaly Detection {dataloader_name} {post_kde_stat}')
    plot_precision_recall_curve(recall, precision, au_prc,
                                title=f'PRC SW Anomaly Detection {dataloader_name} {post_kde_stat}')

    print(f'\tau_roc: {au_roc:.2f}')
    print(f'\tau_prc: {au_prc:.2f}')
