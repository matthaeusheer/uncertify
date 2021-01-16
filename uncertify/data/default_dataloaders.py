"""
A collection of default dataloaders, often used together throughout this project. That way they don't have
to be re-defined in every notebook to keep stuff consistent.
"""
from dataclasses import dataclass
from pathlib import Path

from uncertify.data.dataloaders import dataloader_factory, DatasetType
from uncertify.common import DATA_DIR_PATH, HD_DATA_PATH


@dataclass
class DefaultDataloaderParams:
    batch_size: int = 155
    num_workser: int = 0
    shuffle_val: bool = False


@dataclass
class DefaultDatasetPaths:
    brats_t2_path: Path = HD_DATA_PATH / 'processed/brats17_t2_bc_std_bv3.5.hdf5'
    brats_t2_hm_path: Path = HD_DATA_PATH / 'processed/brats17_t2_hm_bc_std_bv3.5.hdf5'
    brats_t1_path: Path = HD_DATA_PATH / 'processed/brats17_t1_bc_std_bv3.5.hdf5'
    brats_t1_hm_path: Path = HD_DATA_PATH / 'processed/brats17_t1_hm_bc_std_bv-3.5.hdf5'
    camcan_t2_val_path: Path = DATA_DIR_PATH / 'processed/camcan_val_t2_hm_std_bv3.5_xe.hdf5'
    camcan_t2_train_path: Path = DATA_DIR_PATH / 'processed/camcan_train_t2_hm_std_bv3.5_xe.hdf5'
    ibsr_t1_train_path: Path = HD_DATA_PATH / 'processed/ibsr_train_t1_std_bv3.5_l10_xe.hdf5'
    ibsr_t1_val_path: Path = HD_DATA_PATH / 'processed/ibsr_val_t1_std_bv3.5_l10_xe.hdf5'


@dataclass
class DefaultDataloaders:
    brats



_, brats_val_t2_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=BATCH_SIZE, val_set_path=brats_t2_path,
                                                shuffle_val=SHUFFLE_VAL, num_workers=NUM_WORKERS)
_, brats_val_t1_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=BATCH_SIZE, val_set_path=brats_t1_path,
                                                shuffle_val=SHUFFLE_VAL, num_workers=NUM_WORKERS)
_, brats_val_t2_hm_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=BATCH_SIZE,
                                                   val_set_path=brats_t2_hm_path, shuffle_val=SHUFFLE_VAL,
                                                   num_workers=NUM_WORKERS)
_, brats_val_t1_hm_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=BATCH_SIZE,
                                                   val_set_path=brats_t1_hm_path, shuffle_val=SHUFFLE_VAL,
                                                   num_workers=NUM_WORKERS)

_, brats_val_t2_hflip_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=BATCH_SIZE,
                                                      val_set_path=brats_t2_path, shuffle_val=SHUFFLE_VAL,
                                                      num_workers=NUM_WORKERS, transform=H_FLIP_TRANSFORM)
_, brats_val_t2_vflip_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=BATCH_SIZE,
                                                      val_set_path=brats_t2_path, shuffle_val=SHUFFLE_VAL,
                                                      num_workers=NUM_WORKERS, transform=V_FLIP_TRANSFORM)

camcan_train_dataloader, camcan_val_dataloader = dataloader_factory(DatasetType.CAMCAN, batch_size=BATCH_SIZE,
                                                                    val_set_path=camcan_t2_val_path,
                                                                    train_set_path=camcan_t2_train_path,
                                                                    shuffle_val=SHUFFLE_VAL, shuffle_train=True,
                                                                    num_workers=NUM_WORKERS)
camcan_lesional_train_dataloader, camcan_lesional_val_dataloader = dataloader_factory(DatasetType.CAMCAN,
                                                                                      batch_size=BATCH_SIZE,
                                                                                      val_set_path=camcan_t2_val_path,
                                                                                      train_set_path=camcan_t2_train_path,
                                                                                      shuffle_val=False,
                                                                                      shuffle_train=True,
                                                                                      num_workers=NUM_WORKERS,
                                                                                      add_gauss_blobs=True)

noise_loader = DataLoader(noise_set(GaussianNoiseDataset(shape=(1, 128, 128))), batch_size=BATCH_SIZE)

_, mnist_val_dataloader = dataloader_factory(DatasetType.MNIST, batch_size=BATCH_SIZE,
                                             transform=torchvision.transforms.Compose([
                                                 torchvision.transforms.Resize((128, 128)),
                                                 torchvision.transforms.ToTensor()
                                             ])
