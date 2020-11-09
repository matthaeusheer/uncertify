from pathlib import Path

from torch.utils.data import DataLoader
import torchvision

from uncertify.models.vae import load_vae_baur_model
from uncertify.data.dataloaders import DatasetType
from uncertify.data.dataloaders import dataloader_factory
from uncertify.evaluation.evaluation_pipeline import run_evaluation_pipeline
from uncertify.evaluation.configs import EvaluationConfig, PerformanceEvaluationConfig, PixelThresholdSearchConfig
from uncertify.data.datasets import GaussianNoiseDataset
from uncertify.common import DATA_DIR_PATH

# Define some paths and high level parameters
CHECKPOINT_PATH = Path('/media/juniors/2TB_internal_HD/lightning_logs/train_vae/version_3/checkpoints/last.ckpt')
SSD_PROCESSED_DIR_PATH = Path('/home/juniors/code/uncertify/data/processed/')
HDD_PROCESSED_DIR_PATH = Path('/media/juniors/2TB_internal_HD/datasets/processed/')
PROCESSED_DIR_PATH = SSD_PROCESSED_DIR_PATH

DO_SEGMENTATION = False
DO_ANOMALY_DETECTION = True
DO_HISTOGRAMS = True

BATCH_SIZE = 155
USE_N_BATCHES = 2


def main() -> None:
    # Load the model and define the evaluation config
    model = load_vae_baur_model(CHECKPOINT_PATH)
    eval_cfg = EvaluationConfig()
    eval_cfg.use_n_batches = USE_N_BATCHES

    # Define all dataloaders
    _, brats_val_t2_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=BATCH_SIZE,
                                                    val_set_path=HDD_PROCESSED_DIR_PATH / 'brats17_t2_std_bv3.5.hdf5',
                                                    shuffle_val=False)
    _, brats_val_t1_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=BATCH_SIZE,
                                                    val_set_path=HDD_PROCESSED_DIR_PATH / 'brats17_t1_hm_bc_std_bv-3.5.hdf5',
                                                    shuffle_val=False)

    camcan_train_dataloader, camcan_val_dataloader = dataloader_factory(DatasetType.CAMCAN, batch_size=BATCH_SIZE,
                                                                        val_set_path=DATA_DIR_PATH / 'processed/camcan_val_t2_hm_std_bv3.5_xe.hdf5',
                                                                        train_set_path=DATA_DIR_PATH / 'processed/camcan_train_t2_hm_std_bv3.5_xe.hdf5',
                                                                        shuffle_val=False, shuffle_train=True)
    noise_set = GaussianNoiseDataset()
    noise_loader = DataLoader(noise_set, batch_size=BATCH_SIZE)

    _, mnist_val_dataloader = dataloader_factory(DatasetType.MNIST, batch_size=BATCH_SIZE,
                                                 transform=torchvision.transforms.Compose([
                                                     torchvision.transforms.Resize((128, 128)),
                                                     torchvision.transforms.ToTensor()]))

    results = {}

    for dataloader, name in zip([brats_val_t2_dataloader, brats_val_t1_dataloader],
                                ['BraTS T2', 'BraTS T1']):
        result = run_evaluation_pipeline(model,
                                         camcan_train_dataloader,
                                         dataloader,
                                         eval_cfg,
                                         best_threshold=0.9,
                                         run_segmentation=DO_SEGMENTATION,
                                         run_anomaly_detection=DO_ANOMALY_DETECTION,
                                         run_histograms=DO_HISTOGRAMS)
        results[name] = result


if __name__ == '__main__':
    main()
