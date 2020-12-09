import logging
from pathlib import Path

from torch.utils.data import DataLoader
import torchvision

from uncertify.io.models import load_ensemble_models
from uncertify.data.dataloaders import DatasetType
from uncertify.data.dataloaders import dataloader_factory
from uncertify.evaluation.evaluation_pipeline import run_evaluation_pipeline, print_results
from uncertify.evaluation.configs import EvaluationConfig
from uncertify.data.datasets import GaussianNoiseDataset
from uncertify.common import DATA_DIR_PATH
from uncertify.data.dataloaders import MNIST_DEFAULT_TRANSFORM, BRATS_CAMCAN_DEFAULT_TRANSFORM
from uncertify.data.transforms import H_FLIP_TRANSFORM, V_FLIP_TRANSFORM
from uncertify.log import setup_logging

# Define some paths and high level parameters
PROCESSED_DIR_PATH = Path('/media/juniors/2TB_internal_HD/datasets/processed/')

BATCH_SIZE = 155
USE_N_BATCHES = 20
DO_SEGMENTATION = False
DO_ANOMALY_DETECTION = False
DO_LOSS_HISTOGRAMS = True
DO_OOD = True
RESIDUAL_THRESHOLD = 0.95
DO_PLOTS = True
SHUFFLE_VAL = False
LOG = logging.getLogger(__name__)
NUM_WORKERS = 12


def main() -> None:
    # Load models
    RUN_VERSIONS = [1, 2, 3, 4, 5]
    ensemble_models = load_ensemble_models(DATA_DIR_PATH / 'masked_ensemble_models',
                                           [f'model{idx}.ckpt' for idx in RUN_VERSIONS])
    model = ensemble_models[0]

    # Define datasets and loaders
    brats_t2_path = PROCESSED_DIR_PATH / 'brats17_t2_bc_std_bv3.5.hdf5'
    brats_t2_hm_path = PROCESSED_DIR_PATH / 'brats17_t2_hm_bc_std_bv3.5.hdf5'
    brats_t1_path = PROCESSED_DIR_PATH / 'brats17_t1_bc_std_bv3.5.hdf5'
    brats_t1_hm_path = PROCESSED_DIR_PATH / 'brats17_t1_hm_bc_std_bv-3.5.hdf5'
    camcan_t2_val_path = DATA_DIR_PATH / 'processed/camcan_val_t2_hm_std_bv3.5_xe.hdf5'
    camcan_t2_train_path = DATA_DIR_PATH / 'processed/camcan_train_t2_hm_std_bv3.5_xe.hdf5'

    _, brats_val_t2_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=BATCH_SIZE,
                                                    val_set_path=brats_t2_path, shuffle_val=SHUFFLE_VAL)
    _, brats_val_t1_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=BATCH_SIZE,
                                                    val_set_path=brats_t1_path, shuffle_val=SHUFFLE_VAL)
    _, brats_val_t2_hm_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=BATCH_SIZE,
                                                       val_set_path=brats_t2_hm_path, shuffle_val=SHUFFLE_VAL)
    _, brats_val_t1_hm_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=BATCH_SIZE,
                                                       val_set_path=brats_t1_hm_path, shuffle_val=SHUFFLE_VAL)

    camcan_train_dataloader, camcan_val_dataloader = dataloader_factory(DatasetType.CAMCAN, batch_size=BATCH_SIZE,
                                                                        val_set_path=camcan_t2_val_path,
                                                                        train_set_path=camcan_t2_train_path,
                                                                        shuffle_val=SHUFFLE_VAL, shuffle_train=True)
    camcan_lesional_train_dataloader, camcan_lesional_val_dataloader = dataloader_factory(DatasetType.CAMCAN,
                                                                                          batch_size=BATCH_SIZE,
                                                                                          val_set_path=camcan_t2_val_path,
                                                                                          train_set_path=camcan_t2_train_path,
                                                                                          shuffle_val=SHUFFLE_VAL,
                                                                                          shuffle_train=True,
                                                                                          add_gauss_blobs=True)

    _, brats_val_t2_hflip_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=BATCH_SIZE,
                                                          val_set_path=brats_t2_path, shuffle_val=SHUFFLE_VAL,
                                                          num_workers=NUM_WORKERS, transform=H_FLIP_TRANSFORM)
    _, brats_val_t2_vflip_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=BATCH_SIZE,
                                                          val_set_path=brats_t2_path, shuffle_val=SHUFFLE_VAL,
                                                          num_workers=NUM_WORKERS, transform=V_FLIP_TRANSFORM)

    noise_set = GaussianNoiseDataset()
    noise_loader = DataLoader(noise_set, batch_size=BATCH_SIZE)

    _, mnist_val_dataloader = dataloader_factory(DatasetType.MNIST, batch_size=BATCH_SIZE,
                                                 transform=torchvision.transforms.Compose([
                                                     torchvision.transforms.Resize((128, 128)),
                                                     torchvision.transforms.ToTensor()]))

    dataloader_dict = {'BraTS T2 val': brats_val_t2_dataloader,
                       'BraTS T1 val': brats_val_t1_dataloader,
                       'CamCAN lesion val': camcan_lesional_val_dataloader,
                       'BraTS T2 HM val': brats_val_t2_hm_dataloader,
                       'BraTS T1 HM val': brats_val_t1_hm_dataloader,
                       'Gaussian noise': noise_loader,
                       'MNIST': mnist_val_dataloader,
                       'BraTS T2 HFlip': brats_val_t2_hflip_dataloader,
                       'BraTS T2 VFlip': brats_val_t2_vflip_dataloader
                       }
    for name, dataloader in dataloader_dict.items():
        print(f'{name:15} dataloader: {len(dataloader)} batches (batch_size: {dataloader.batch_size}) '
              f'-> {len(dataloader) * dataloader.batch_size} samples.')

    eval_cfg = EvaluationConfig()
    eval_cfg.do_plots = DO_PLOTS
    eval_cfg.use_n_batches = USE_N_BATCHES
    eval_cfg.ood_config.metrics = ('waic', 'dose')  # 'dose'
    eval_cfg.ood_config.dose_statistics = ('rec_err', 'kl_div', 'elbo')  # 'entropy'

    results = {}
    counter = 0
    for name, dataloader in dataloader_dict.items():
        LOG.info(f'Running evaluation on {name}!')
        result = run_evaluation_pipeline(model,
                                         camcan_train_dataloader,
                                         'CamCAN T2',
                                         dataloader,
                                         name,
                                         eval_cfg,
                                         RESIDUAL_THRESHOLD,
                                         run_segmentation=DO_SEGMENTATION,
                                         run_anomaly_detection=DO_ANOMALY_DETECTION,
                                         run_loss_histograms=DO_LOSS_HISTOGRAMS,
                                         run_ood_detection=DO_OOD,
                                         ensemble_models=ensemble_models)
        results[name] = result
        counter += 1

    for name, result in results.items():
        print(f'\n\t{name}')
        print_results(result)


if __name__ == '__main__':
    setup_logging()
    main()
