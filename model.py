import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from segmentation_metric import *
from utils import *


class UNETDoubleConv(nn.Module):
    """
    Double Convolution and BN and ReLU
    (3x3 conv -> BN -> ReLU) ** 2
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNETDown(nn.Module):
    """
    Combination of MaxPool2d and DoubleConv in series
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), UNETDoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class UNETUp(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path,
    followed by double 3x3 convolution.
    """

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(
                in_ch, in_ch // 2, kernel_size=2, stride=2
            )

        self.conv = UNETDoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(
            x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2]
        )

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNET(nn.Module):
    """
    Architecture based on U-Net: Convolutional Networks for Biomedical Image Segmentation
    Link - https://arxiv.org/abs/1505.04597

    Parameters:
        num_classes: Number of output classes required (default 19 for KITTI dataset)
        num_layers: Number of layers in each side of U-net
        features_start: Number of features in first layer
        bilinear: Whether to use bilinear interpolation or transposed
            convolutions for upsampling.
    """

    def __init__(
        self,
        num_classes: int = 19,
        num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers

        layers = [UNETDoubleConv(3, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(UNETDown(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(UNETUp(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1 : self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers : -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])


class UNETModel(pl.LightningModule):
    def __init__(self, config, logger):
        super(UNETModel, self).__init__()

        self.config = config
        self.n_logger = logger

        self.model = UNET(
            num_classes=self.config["num_classes"],
            num_layers=self.config["num_layers"],
            features_start=self.config["features_start"],
            bilinear=self.config["bilinear"],
        )

    def forward(self, x):
        return self.model(x)

    def poly_lr_scheduler(self, epoch, num_epochs=300, power=0.9):
        return (1 - epoch / num_epochs) ** power

    def configure_optimizers(self):

        if self.config["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                betas=self.config["beta"],
                eps=self.config["eps"],
                weight_decay=self.config["weight_decay"],
                amsgrad=self.config["amsgrad"],
            )
        elif self.config["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                momentum=self.config["momentum"],
                weight_decay=self.config["weight_decay"],
            )

        if self.config["use_scheduler"]:

            if self.config["scheduler"] == "MultiStepLR":
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[
                        int(
                            self.config["MultiStepLR_milestones_1"]
                            * self.config["max_epochs"]
                        ),
                        int(
                            self.config["MultiStepLR_milestones_2"]
                            * self.config["max_epochs"]
                        ),
                    ],
                    gamma=self.config["MultiStepLR_gamma"],
                )

            elif self.config["scheduler"] == "LambdaLR":
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=self.poly_lr_scheduler(
                        num_epochs=self.config["max_epochs"]
                    ),
                )

            elif self.config["scheduler"] == "CyclicLR":
                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=self.config["CyclicLR_base_lr"],
                    max_lr=self.config["CyclicLR_max_lr"],
                    step_size_up=self.config["CyclicLR_step_size_up"],
                    mode=self.config["CyclicLR_mode"],
                    cycle_momentum=self.config["CyclicLR_cycle_momentum"],
                )

            elif self.config["scheduler"] == "OneCyclicLR":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=0.1,
                    steps_per_epoch=10,
                    epochs=self.config["max_epochs"],
                )

            elif self.config["scheduler"] == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=self.config["ReduceLROnPlateau_mode"],
                    factor=self.config["ReduceLROnPlateau_factor"],
                    patience=self.config["ReduceLROnPlateau_patience"],
                    verbose=True,
                )

            elif self.config["scheduler"] == "CosineAnnealingWarmRestarts":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=self.config["CosineAnnealingWarmRestarts_T_0"],
                    T_mult=self.config["CosineAnnealingWarmRestarts_T_mult"],
                    eta_min=self.config["CosineAnnealingWarmRestarts_eta_min"],
                    last_epoch=self.config["CosineAnnealingWarmRestarts_last_epoch"],
                )

            return [optimizer], [scheduler]

        else:
            return optimizer

    def loss_function(self, logits, labels):

        loss = F.cross_entropy(logits, labels, ignore_index=255)
        return loss

    def training_step(self, batch, batch_idx):

        images, masks, labels = batch

        if masks.type() == "torch.cuda.FloatTensor":
            masks = torch.round(masks * 255).long()
        else:
            masks = masks.long()

        if len(masks.shape) == 4:
            masks = masks.squeeze().long()

        output = self.forward(images)
        # print("output: {}".format(output.shape))

        loss = self.loss_function(output, masks)

        train_prediction = output.argmax(1)

        # Calculate the Dice coefficents without background
        (
            train_dice_list,
            train_dice_indicator,
        ) = SegmentationEvaluation.dice_wobg_pytorch(
            train_prediction.detach(),
            masks.detach(),
            device=self.device,
            n_classes=self.config["num_classes"],
        )

        # Calculate the mean Dice
        train_mDice = (
            train_dice_list * train_dice_indicator
        ).sum() / train_dice_indicator.sum()
        self.log(
            "train_mDice", train_mDice, on_step=False, on_epoch=True, sync_dist=True
        )

        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):

        images, masks, labels = batch

        if masks.type() == "torch.cuda.FloatTensor":
            masks = torch.round(masks * 255).long()
        else:
            masks = masks.long()

        if len(masks.shape) == 4:
            masks = masks.squeeze().long()

        output = self.forward(images)

        val_loss = self.loss_function(output, masks)

        val_prediction = output.argmax(1)

        # Calculate the Dice coefficents without background
        val_dice_list, val_dice_indicator = SegmentationEvaluation.dice_wobg_pytorch(
            val_prediction.detach(),
            masks.detach(),
            device=self.device,
            n_classes=self.config["num_classes"],
        )

        # Calculate the mean Dice
        val_mDice = (
            val_dice_list * val_dice_indicator
        ).sum() / val_dice_indicator.sum()
        self.log("val_mDice", val_mDice, on_step=False, on_epoch=True, sync_dist=True)

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, sync_dist=True)

        if self.current_epoch % self.config["fig_save_interval"] == 0:

            # plot the results (segmentation figures) to pdf files
            dict_single_batch = Segmentation.generate_dict_single_batch(
                self.config, images, output, masks
            )

            if self.config["part_results_saving"]:

                if batch_idx == 0:
                    self.val_dict_list = []
                    self.val_dict_list.append(dict_single_batch)

                if batch_idx < self.config["num_saved_batches"] and batch_idx > 0:
                    self.val_dict_list.append(dict_single_batch)
                elif batch_idx == self.config["num_saved_batches"]:
                    self.val_dict_list.append(dict_single_batch)
                    Segmentation.plot2file_from_dict_list(
                        self.config,
                        self.val_dict_list,
                        self.device,
                        self.current_epoch,
                        self.config["num_saved_batches"],
                    )

            else:

                if batch_idx == 0:
                    self.val_dict_list = []
                    self.val_dict_list.append(dict_single_batch)

                if (
                    batch_idx
                    < (self.config["num_batches_list"][1] // self.config["gpus"])
                    and batch_idx > 0
                ):
                    self.val_dict_list.append(dict_single_batch)
                elif batch_idx == (
                    self.config["num_batches_list"][1] // self.config["gpus"]
                ):
                    self.val_dict_list.append(dict_single_batch)
                    Segmentation.plot2file_from_dict_list(
                        self.config, self.val_dict_list, self.device, self.current_epoch
                    )

    def test_step(self, batch, batch_idx):

        images, masks, labels = batch

        if masks.type() == "torch.cuda.FloatTensor":
            masks = torch.round(masks * 255).long()
        else:
            masks = masks.long()

        if len(masks.shape) == 4:
            masks = masks.squeeze().long()

        out = self.forward(images)

        output_out = F.interpolate(
            out, size=images.size()[-2:], mode="bilinear", align_corners=True
        )

        test_loss = self.loss_function(output_out, masks)

        test_prediction = output_out.argmax(1)

        test_dice_list, test_dice_indicator = SegmentationEvaluation.dice_wobg_pytorch(
            test_prediction.detach(),
            masks.detach(),
            device=self.device,
            n_classes=self.config["num_classes"],
        )

        # Calculate the mean Dice
        test_mDice = (
            test_dice_list * test_dice_indicator
        ).sum() / test_dice_indicator.sum()
        self.log("test_mDice", test_mDice, on_step=False, on_epoch=True, sync_dist=True)

        self.log("test_loss", test_loss, on_step=False, on_epoch=True, sync_dist=True)

        if self.current_epoch % self.config["fig_save_interval"] == 0:

            # plot the results (segmentation figures) to pdf files
            dict_single_batch = Segmentation.generate_dict_single_batch(
                self.config, images, output_out, masks
            )

            if self.config["part_results_saving"]:

                if batch_idx == 0:
                    self.test_dict_list = []
                    self.test_dict_list.append(dict_single_batch)

                if batch_idx < self.config["num_saved_batches"] and batch_idx > 0:
                    self.test_dict_list.append(dict_single_batch)
                elif batch_idx == self.config["num_saved_batches"]:
                    self.test_dict_list.append(dict_single_batch)
                    Segmentation.plot2file_from_dict_list(
                        self.config,
                        self.test_dict_list,
                        self.device,
                        self.current_epoch,
                        self.config["num_saved_batches"],
                    )

            else:

                if batch_idx == 0:
                    self.test_dict_list = []
                    self.test_dict_list.append(dict_single_batch)

                if (
                    batch_idx
                    < (self.config["num_batches_list"][1] // self.config["gpus"])
                    and batch_idx > 0
                ):
                    self.test_dict_list.append(dict_single_batch)
                elif batch_idx == (
                    self.config["num_batches_list"][1] // self.config["gpus"]
                ):
                    self.test_dict_list.append(dict_single_batch)
                    Segmentation.plot2file_from_dict_list(
                        self.config,
                        self.test_dict_list,
                        self.device,
                        self.current_epoch,
                    )
