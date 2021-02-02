import numpy as np

from uncertify.evaluation.model_performance import calculate_roc, calculate_prc
from uncertify.visualization.model_performance import plot_roc_curve, plot_precision_recall_curve
from uncertify.visualization.histograms import plot_multi_histogram


def slice_wise_lesion_detection_dose_kde(dataloader_dict: dict, metrics_ood_dict: dict, dose_statistics: list,
                                         do_plots: bool = True) -> None:
    """For a given metrics_ood_dict, check if the dose_kde metrics can be used for slice-wise anomaly detection.

    Arguments:
        dataloader_dict: a (name, dataloader) key, value dictionary
        metrics_ood_dict: a dictionary es returned by run_ood_to_ood_dict
        dose_statistics: a list of dose statistics we take into account here, e.g. entropy, kl_div, rec_err, elbo
        do_plots: if yes, will to roc, prc and histogram plots
    """
    for dataloader_name in dataloader_dict.keys():
        if 'can' in dataloader_name.lower():
            continue
        print()
        print(f' --- {dataloader_name} --- ')
        # post_kde_stat are statistics names we use for the dose_kde
        for post_kde_stat in list(dose_statistics) + ['_'.join(dose_statistics)]:
            print(f'{post_kde_stat}')
            # all dose statistics combined
            if post_kde_stat == '_'.join(dose_statistics):  # needs correct ordering, not so nice...
                healthy_post_kde = metrics_ood_dict['dose'][dataloader_name]['healthy']
                lesional_post_kde = metrics_ood_dict['dose'][dataloader_name]['lesional']
            # individual statistics
            else:
                healthy_post_kde = metrics_ood_dict['dose'][dataloader_name]['dose_kde_healthy'][post_kde_stat]
                lesional_post_kde = metrics_ood_dict['dose'][dataloader_name]['dose_kde_lesional'][post_kde_stat]
            # print(f'{len(healthy_post_kde)} healthy / {len(lesional_post_kde)} lesional samples')

            y_true = len(healthy_post_kde) * [0.0] + len(lesional_post_kde) * [1.0]
            y_pred_proba = healthy_post_kde + lesional_post_kde

            fpr, tpr, _, au_roc = calculate_roc(y_true, np.array(y_pred_proba))
            precision, recall, _, au_prc = calculate_prc(y_true, np.array(y_pred_proba))

            if do_plots:
                plot_roc_prc_slice_wise_lesion_detection_dose_kde(dataloader_name, post_kde_stat, fpr, tpr, au_roc,
                                                                  recall, precision, au_prc,
                                                                  healthy_post_kde, lesional_post_kde)

            # yield fpr, tpr, au_roc, recall, precision, au_prc, healthy_post_kde, lesional_post_kde


def plot_roc_prc_slice_wise_lesion_detection_dose_kde(dataloader_name, post_kde_stat, fpr, tpr, au_roc, recall,
                                                      precision, au_prc, healthy_post_kde,
                                                      lesional_post_kde):
    plot_multi_histogram([np.array(healthy_post_kde), np.array(lesional_post_kde)],
                         ['healthy', 'lesional'],
                         plot_density=False,
                         show_data_ticks=False,
                         legend_title=f'{dataloader_name} ' + '$DoSE_{KDE}$' + f'({post_kde_stat})',
                         legend_pos='upper left')
    plot_roc_curve(fpr, tpr, au_roc, title=f'ROC SW Anomaly Detection {dataloader_name} {post_kde_stat}')
    plot_precision_recall_curve(recall, precision, au_prc, title=f'PRC SW Anomaly Detection {dataloader_name} {post_kde_stat}')

    print(f'\tau_roc: {au_roc:.2f}')
    print(f'\tau_prc: {au_prc:.2f}')
