import inspect
import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as transforms
from .data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, TestDatasetFromFolder, TestDatasetFromFolder2, TestDatasetFromFolderinType

class DInterface(pl.LightningDataModule):

    def __init__(self, params, num_workers=8,
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.kwargs = kwargs
        self.batch_size = params.batch_size
        self.params = params
        # self.load_data_module()

    def setup(self, stage=None):
        train_dataset = TrainDatasetFromFolder(
            dataset_dir=self.params.train_dataset_dir,
            original_size=self.params.original_size,
            crop_size=self.params.crop_size,
            upscale_factor=self.params.upscale_factor
        )
        val_dataset = ValDatasetFromFolder(
            dataset_dir=self.params.val_dataset_dir,
            crop_size=self.params.crop_size,
            upscale_factor=self.params.upscale_factor
        )
        test_dataset = TestDatasetFromFolder2(
            dataset_dir=self.params.test_dataset_dir,
            crop_size=self.params.crop_size,
            original_size=self.params.original_size,
            upscale_factor=self.params.upscale_factor
        )
        # for class test
        # test_dataset = TestDatasetFromFolderinType(
        #     dataset_dir=self.params.test_dataset_dir,
        #     type_name=self.params.type_name,
        #     crop_size=self.params.crop_size,
        #     original_size=self.params.original_size,
        #     upscale_factor=self.params.upscale_factor,
        #     type_name = self.params.type_name,
        #     n_augmentations=self.params.n_augmentations,
        #     augment=self.params.augment,
        #     inference_mode=self.params.inference_mode
        # )

        self.datasets = {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        }

    def train_dataloader(self):
        train_dataset = self.datasets["train"]
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )


    def val_dataloader(self, shuffle=False):
        val_dataset = self.datasets["validation"]
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self, shuffle=False):
        test_dataset = self.datasets["test"]
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle
        )