import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PascalVOC(torch.utils.data.Dataset):
    CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted-plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
        "ambiguous",
    ]

    CLASS_IDX = {
        "background": 0,
        "aeroplane": 1,
        "bicycle": 2,
        "bird": 3,
        "boat": 4,
        "bottle": 5,
        "bus": 6,
        "car": 7,
        "cat": 8,
        "chair": 9,
        "cow": 10,
        "diningtable": 11,
        "dog": 12,
        "horse": 13,
        "motorbike": 14,
        "person": 15,
        "potted-plant": 16,
        "sheep": 17,
        "sofa": 18,
        "train": 19,
        "tv/monitor": 20,
        "ambiguous": 255,
    }

    CLASS_IDX_INV = {
        0: "background",
        1: "aeroplane",
        2: "bicycle",
        3: "bird",
        4: "boat",
        5: "bottle",
        6: "bus",
        7: "car",
        8: "cat",
        9: "chair",
        10: "cow",
        11: "diningtable",
        12: "dog",
        13: "horse",
        14: "motorbike",
        15: "person",
        16: "potted-plant",
        17: "sheep",
        18: "sofa",
        19: "train",
        20: "tv/monitor",
        255: "ambiguous",
    }

    NUM_CLASS = 21

    # MEAN = (0.485, 0.456, 0.406)
    # STD = (0.229, 0.224, 0.225)

    def __init__(self, config):
        super().__init__()

        self.config = config

        assert self.NUM_CLASS == self.config["num_classes"], print(
            "The number of classes does not match."
        )

        self._init_palette()

        self.train_transform = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=self.config["crop_size"],
                    min_width=self.config["crop_size"],
                ),
                A.RandomCrop(self.config["crop_size"], self.config["crop_size"]),
                A.Flip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=self.config["mean"], std=self.config["std"]),
                ToTensorV2(),
            ]
        )

        self.test_transform = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=self.config["crop_size"],
                    min_width=self.config["crop_size"],
                ),
                A.CenterCrop(
                    p=1, height=self.config["crop_size"], width=self.config["crop_size"]
                ),
                A.Flip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=self.config["mean"], std=self.config["std"]),
                ToTensorV2(),
            ]
        )

    def colormap(self, N=256):
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        dtype = "uint8"
        cmap = []
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap.append((r, g, b))

        return cmap

    def _init_palette(self):
        from PIL import Image, ImagePalette

        self.cmap = self.colormap()
        self.palette = ImagePalette.ImagePalette()
        for rgb in self.cmap:
            self.palette.getcolor(rgb)

    def get_palette(self):
        return self.palette

    def denorm(self, image):

        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [CxHxW]"
            assert image.size(0) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip(image, self.MEAN, self.STD):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            # batch mode
            assert image.size(1) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip((0, 1, 2), self.MEAN, self.STD):
                image[:, t, :, :].mul_(s).add_(m)

        return image


class PascalVOCClsSeg(PascalVOC):
    def __init__(self, config, split, root=os.path.expanduser("../datasets/")):

        super(PascalVOCClsSeg, self).__init__(config)

        self.config = config

        self.root = root
        self.split = split

        # train/val/test splits are pre-cut
        if self.split == "train":
            _split_f = os.path.join(self.root, "train_augvoc.txt")
            self.transform = self.train_transform
        elif self.split == "train_voc":
            _split_f = os.path.join(self.root, "train_voc.txt")
            self.transform = self.train_transform
        elif self.split == "val":
            _split_f = os.path.join(self.root, "val_voc.txt")
            self.transform = self.test_transform
        elif self.split == "test":
            _split_f = os.path.join(self.root, "test.txt")
            self.transform = self.test_transform
        else:
            raise RuntimeError("Unknown dataset split.")

        assert os.path.isfile(_split_f), "%s not found" % _split_f

        self.images = []
        self.masks = []
        with open(_split_f, "r") as lines:
            for line in lines:
                _image, _mask = line.strip("\n").split(" ")
                _image = os.path.join(self.root, _image)
                assert os.path.isfile(_image), "%s not found" % _image
                self.images.append(_image)

                if self.split != "test":
                    _mask = os.path.join(self.root, _mask.lstrip("/"))
                    assert os.path.isfile(_mask), "%s not found" % _mask
                    self.masks.append(_mask)

        if self.split != "test":
            assert len(self.images) == len(self.masks)
            if self.split == "train":
                assert len(self.images) == 10582
            elif self.split == "val":
                assert len(self.images) == 1449

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = np.array(Image.open(self.images[index]).convert("RGB"))
        mask = np.array(Image.open(self.masks[index]))
        # print("image shape: {}".format(image.shape))
        # print("mask shape: {}".format(mask.shape))

        unique_labels = np.unique(mask)

        # ambigious
        if unique_labels[-1] == self.CLASS_IDX["ambiguous"]:
            unique_labels = unique_labels[:-1]

        labels = torch.zeros(self.NUM_CLASS - 1)
        if unique_labels[0] == self.CLASS_IDX["background"]:
            unique_labels = unique_labels[1:]
        unique_labels -= 1  # shifting since no BG class

        assert unique_labels.size > 0, "No labels found in %s" % self.masks[index]
        labels[unique_labels.tolist()] = 1

        # general resize, normalize and toTensor
        # image, mask = self.transform(image, mask)
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        # return (image, segmentation_mask, classification_label)
        return image, mask.long(), labels

    @property
    def pred_offset(self):
        return 0


class PascalVOCDataset(PascalVOC):
    def __init__(self, config, validation=None, cross_validation=None, num_folds=None):

        super(PascalVOCDataset, self).__init__(config)

        self.config = config

        self.validation = validation
        self.cross_validation = cross_validation

        if self.cross_validation:
            self.num_folds = self.config["num_folds"]

        self.config["data_root"] = os.path.join(
            self.config["data_download_root"], self.config["dataset_name"]
        )

        self.train_dataset, self.val_dataset, self.test_dataset = self.get_datasets()
        print(
            "len train_dataset: {}, len val_dataset: {}, len test_dataset: {}".format(
                len(self.train_dataset), len(self.val_dataset), len(self.test_dataset)
            )
        )

    def get_datasets(self):

        train_dataset = PascalVOCClsSeg(
            self.config, "train", root=self.config["data_root"]
        )

        val_dataset = PascalVOCClsSeg(
            self.config, "train_voc", root=self.config["data_root"]
        )

        test_dataset = PascalVOCClsSeg(
            self.config, "val", root=self.config["data_root"]
        )

        return train_dataset, val_dataset, test_dataset

    def get_dataloaders(self):

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            drop_last=True,
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"],
        )

        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            drop_last=False,
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"],
        )

        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            drop_last=False,
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"],
        )

        if self.cross_validation:
            print("-*10 Cross Validation is not used -*10")

        if self.validation:

            return train_dataloader, val_dataloader, test_dataloader
        else:
            return train_dataloader, test_dataloader


if __name__ == "__main__":
    import warnings

    from rich import print

    warnings.filterwarnings("ignore")

    from expr_setting import *

    experiment_name = "config"

    experiment_setting = ExprSetting(experiment_name)
    prepare_dataset = PrepareDataset()

    experiment_config = experiment_setting.config
    print("experiment_config: {}".format(experiment_config))

    dataset_class = prepare_dataset.get_dataset_class(experiment_config)

    if experiment_config["cross_validation"]:
        dataset_instance = dataset_class(
            experiment_config,
            cross_validation=True,
            num_fold=experiment_config["num_folds"],
        )
        (
            trian_dataloader_list,
            val_dataloader_list,
        ) = dataset_instance.get_dataloaders()
        print("trian_dataloader_list: {}".format(len(trian_dataloader_list)))
        print("val_dataloader_list: {}".format(len(val_dataloader_list)))

        for fold_index in range(experiment_config["num_fold"]):
            train_dataloader = trian_dataloader_list[fold_index]
            val_dataloader = val_dataloader_list[fold_index]

    else:
        if experiment_config["validation"]:
            dataset_instance = dataset_class(experiment_config, validation=True)
            (
                train_dataloader,
                val_dataloader,
                test_dataloader,
            ) = dataset_instance.get_dataloaders()
            print(
                "train_dataloader: {}, val_dataloader: {}, test_dataloader: {}".format(
                    len(train_dataloader), len(val_dataloader), len(test_dataloader)
                )
            )

        else:
            dataset_instance = dataset_class(experiment_config)
            train_dataloader, test_dataloader = dataset_instance.get_dataloaders()
            print(
                "train_dataloader: {}, test_dataloader: {}".format(
                    len(train_dataloader), len(test_dataloader)
                )
            )

    for batch_idx, (image, mask, label) in enumerate(test_dataloader):
        print(
            "batch_idx: {}, image: {}, mask: {}, label: {}".format(
                batch_idx, image.shape, mask.shape, label.shape
            )
        )
