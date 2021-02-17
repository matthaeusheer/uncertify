import logging
import numpy as np

from uncertify.evaluation.model_performance import calculate_roc, calculate_prc
from uncertify.visualization.model_performance import plot_roc_curve, plot_precision_recall_curve
from uncertify.visualization.histograms import plot_multi_histogram
from uncertify.visualization.plotting import save_fig
from uncertify.common import DEFAULT_PLOT_DIR_PATH

# from uncertify.visualization.dose import LABEL_MAP

LABEL_MAP = {
    'rec_err': '$\ell_{1}$',
    'kl_div': '$D_{KL}$',
    'elbo': '$\mathcal{L}$',
    'entropy': '$H_{\ell_{1}}$',
    'entropy_rec_err_kl_div_elbo': '$H_{\ell_{1}}, \ell_{1}, D_{KL}, \mathcal{L}',
    'waic': '$WAIC$'
}

LOG = logging.getLogger(__name__)


def slice_wise_lesion_detection_dose_kde(dataloader_dict: dict, train_loader_name: str,
                                         metrics_ood_dict: dict, dose_statistics: list, predict_mode: str,
                                         do_plots: bool = True, save_figs: bool = True, show_title: bool = True,
                                         show_legend: bool = True) -> None:
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
        if 'can' in dataloader_name.lower() and 'lesion' not in dataloader_name.lower():
            continue  # no need to do slice-wise lesion detection here
        print()
        print(f' --- {dataloader_name} --- ')
        # post_kde_stat are statistics names we use for the dose_kde
        joined_stats_name = '_'.join(dose_statistics)
        for stat_name in list(dose_statistics) + [joined_stats_name]:
            kde_in_dist = metrics_ood_dict['dose'][train_loader_name]['dose_kde_healthy'][stat_name]
            stat_in_dist = metrics_ood_dict['dose'][train_loader_name]['dose_stat_healthy'][stat_name]
            print(f'{stat_name}')
            stat_healthy_ood, stat_lesional_ood = None, None
            # all dose statistics combined
            if stat_name == joined_stats_name:
                kde_healthy_ood = metrics_ood_dict['dose'][dataloader_name]['healthy']
                kde_lesional_ood = metrics_ood_dict['dose'][dataloader_name]['lesional']
                kde_in_dist = metrics_ood_dict['dose'][train_loader_name]['healthy']
                stat_in_dist = None
            # individual statistics
            else:
                kde_healthy_ood = metrics_ood_dict['dose'][dataloader_name]['dose_kde_healthy'][stat_name]
                kde_lesional_ood = metrics_ood_dict['dose'][dataloader_name]['dose_kde_lesional'][stat_name]
                stat_healthy_ood = metrics_ood_dict['dose'][dataloader_name]['dose_stat_healthy'][stat_name]
                stat_lesional_ood = metrics_ood_dict['dose'][dataloader_name]['dose_stat_lesional'][stat_name]
                # print(f'\t{len(kde_healthy_ood)} healthy & {len(kde_lesional_ood)} lesional samples')

            n_healthy_ood = len(kde_healthy_ood)
            n_lesional_ood = len(kde_lesional_ood)
            n_healthy_id = n_healthy_ood

            y_true = n_healthy_id * [0.0] + n_healthy_ood * [0.0] + n_lesional_ood * [1.0]

            kde_in_dist_samples = list(np.random.choice(kde_in_dist, size=n_healthy_id, replace=False))
            if predict_mode == 'kde':
                y_pred_proba = np.array(kde_in_dist_samples + kde_healthy_ood + kde_lesional_ood)
                # For DoSE KDE scores high scores should reflect not lesional, so invert scorings
                y_pred_proba *= -1.0
            elif predict_mode == 'stat':
                try:
                    stat_in_dist = list(np.random.choice(stat_in_dist, size=n_healthy_id, replace=False))
                except ValueError:
                    LOG.warning(f'Ignored {stat_name}.')
                    continue
                y_pred_proba = np.array(stat_in_dist + stat_healthy_ood + stat_lesional_ood)
            else:
                raise ValueError('Predict mode not supported.')

            fpr, tpr, _, au_roc = calculate_roc(y_true, y_pred_proba)
            precision, recall, _, au_prc = calculate_prc(y_true, y_pred_proba)

            if do_plots:
                plot_roc_prc_slice_wise_lesion_detection_dose_kde(
                    train_loader_name, dataloader_name, stat_name,
                    fpr, tpr, au_roc,
                    recall, precision, au_prc,
                    kde_in_dist_samples, kde_healthy_ood, kde_lesional_ood,
                    stat_in_dist, stat_healthy_ood, stat_lesional_ood,
                    save_figs, show_legend, show_title
                )


def slice_wise_lesion_detection_waic(dataloader_dict: dict, train_loader_name: str, metrics_ood_dict: dict,
                                     do_plots: bool = True, save_figs: bool = True, show_title: bool = True,
                                     show_legend: bool = True) -> None:
    for dataloader_name in dataloader_dict.keys():
        if 'can' in dataloader_name.lower() and 'lesion' not in dataloader_name.lower():
            continue  # no need to do slice-wise lesion detection here
        print()
        print(f' --- {dataloader_name} --- ')
        healthy_in_dist = metrics_ood_dict['waic'][train_loader_name]['healthy']
        healthy_ood = metrics_ood_dict['waic'][dataloader_name]['healthy']
        lesional_ood = metrics_ood_dict['waic'][dataloader_name]['lesional']

        n_healthy_ood = len(healthy_ood)
        n_lesional_ood = len(lesional_ood)
        n_healthy_id = n_healthy_ood

        y_true = n_healthy_id * [0.0] + n_healthy_ood * [0.0] + n_lesional_ood * [1.0]
        healthy_in_dist_samples = list(np.random.choice(healthy_in_dist, size=n_healthy_id, replace=False))
        y_pred_proba = np.array(healthy_in_dist_samples + healthy_ood + lesional_ood)

        fpr, tpr, _, au_roc = calculate_roc(y_true, y_pred_proba)
        precision, recall, _, au_prc = calculate_prc(y_true, y_pred_proba)

        if do_plots:
            plot_roc_prc_slice_wise_lesion_detection_dose_kde(
                train_loader_name, dataloader_name, 'waic',
                fpr, tpr, au_roc,
                recall, precision, au_prc,
                healthy_in_dist_samples, healthy_ood, lesional_ood,
                id_healthy_stat=None, healthy_stat=None, lesional_stat=None,
                save_figs=True, show_legend=True, show_title=True
            )


def plot_roc_prc_slice_wise_lesion_detection_dose_kde(train_loader_name, test_loader_name, post_kde_stat,
                                                      fpr, tpr, au_roc, recall, precision, au_prc,
                                                      healthy_id_kde, healthy_post_kde, lesional_post_kde,
                                                      id_healthy_stat=None, healthy_stat=None, lesional_stat=None,
                                                      save_figs: bool = True, show_legend: bool = True,
                                                      show_title: bool = True):
    kde_hist_fig_title = f'{train_loader_name}_{test_loader_name}_kde_hist_{post_kde_stat}'
    array_labels = None if not show_legend else [f'{train_loader_name} healthy',
                                                 f'{test_loader_name} healthy OOD',
                                                 f'{test_loader_name} lesional OOD']
    if post_kde_stat == 'waic':
        xlabel = 'waic'
    else:
        xlabel = '$DoSE_{KDE}($' + LABEL_MAP[post_kde_stat] + ')'
    kde_hist_fig, _ = plot_multi_histogram(
        [np.array(healthy_id_kde), np.array(healthy_post_kde), np.array(lesional_post_kde)],
        array_labels,
        plot_density=False,
        show_data_ticks=False,
        legend_pos='upper left',
        figsize=(3, 3),
        xlabel=xlabel)
    if save_figs:
        save_fig(kde_hist_fig, DEFAULT_PLOT_DIR_PATH / f'{kde_hist_fig_title}.png')

    if healthy_stat is not None and lesional_stat is not None and id_healthy_stat is not None:
        stat_hist_fig_title = f'{train_loader_name}_{test_loader_name}_stat_hist_{post_kde_stat}'
        stat_hist_fig, _ = plot_multi_histogram(
            [np.array(id_healthy_stat), np.array(healthy_stat), np.array(lesional_stat)],
            array_labels,
            plot_density=False,
            show_data_ticks=False,
            legend_pos='upper left',
            figsize=(3, 3),
            xlabel=LABEL_MAP[post_kde_stat])
        if save_figs:
            save_fig(stat_hist_fig, DEFAULT_PLOT_DIR_PATH / f'{stat_hist_fig_title}.png')

    roc_fig = plot_roc_curve(
        fpr, tpr, au_roc,
        title=f'ROC SW Anomaly Detection {test_loader_name} {post_kde_stat}' if show_title else None,
        figsize=(3, 3))
    if save_figs:
        save_fig(roc_fig, DEFAULT_PLOT_DIR_PATH / f'roc_sw_{test_loader_name}_{post_kde_stat}')

    prc_fig = plot_precision_recall_curve(
        recall, precision, au_prc,
        title=f'PRC SW Anomaly Detection {test_loader_name} {post_kde_stat}' if show_title else None,
        figsize=(3, 3))
    if save_figs:
        save_fig(prc_fig, DEFAULT_PLOT_DIR_PATH / f'prc_sw_{test_loader_name}_{post_kde_stat}')

    print(f'\tau_roc: {au_roc:.2f}')
    print(f'\tau_prc: {au_prc:.2f}')
