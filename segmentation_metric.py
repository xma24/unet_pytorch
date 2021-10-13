import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("ggplot")

import warnings

warnings.filterwarnings("ignore")

from utils import UnNormalize

SMOOTH = 1e-6


class SegmentationEvaluation(object):
    @classmethod
    def iou_pytorch_wobg(
        self,
        pred,
        target,
        device=None,
        n_classes=12,
    ):
        # both pred and target are tensors;
        # each value in pred and target shoud be the class id, not one-hot tensor;
        ious = []
        iou_indicator = []
        pred = pred.view(-1)
        target = target.view(-1)

        # do not calculate the mIoU of background
        for cls in range(1, n_classes):
            # cls is integer from 1 to n_classes-1,
            # This goes from 1:n_classes-1 -> class "0" is ignored
            pred_inds = pred == cls
            target_inds = target == cls
            intersection = (pred_inds[target_inds]).long().sum().item()

            # print("intersection: {}".format(intersection))

            # Cast to long to prevent overflows
            union = (
                pred_inds.long().sum().item()
                + target_inds.long().sum().item()
                - intersection
            )
            # print("union: {}".format(union))

            if union == 0:

                ious.append(torch.tensor(0.0))
                iou_indicator.append(torch.tensor(0.0))

            # If there is no ground truth, do not include in evaluation
            else:
                ious.append(torch.tensor(float(intersection) / float(max(union, 1))))
                iou_indicator.append(torch.tensor(1.0))
        # print("ious: {}".format(ious))

        ious = torch.as_tensor(ious)
        iou_indicator = torch.as_tensor(iou_indicator)

        if device:
            ious = ious.to(device)
            iou_indicator = iou_indicator.to(device)

        return ious, iou_indicator

    @classmethod
    def iou_with_bg(
        self,
        pred,
        target,
        device=None,
        n_classes=12,
    ):
        # both pred and target are tensors;
        # each value in pred and target shoud be the class id, not one-hot tensor;
        ious = []
        iou_indicator = []
        pred = pred.view(-1)
        target = target.view(-1)

        # calculate the mIoU of background
        for cls in range(n_classes):
            pred_inds = pred == cls
            target_inds = target == cls
            intersection = (pred_inds[target_inds]).long().sum().item()

            # print("intersection: {}".format(intersection))

            # Cast to long to prevent overflows
            union = (
                pred_inds.long().sum().item()
                + target_inds.long().sum().item()
                - intersection
            )
            # print("union: {}".format(union))

            if union == 0:

                ious.append(torch.tensor(0.0))
                iou_indicator.append(torch.tensor(0.0))

            # If there is no ground truth, do not include in evaluation
            else:
                ious.append(torch.tensor(float(intersection) / float(max(union, 1))))
                iou_indicator.append(torch.tensor(1.0))
        # print("ious: {}".format(ious))

        ious = torch.as_tensor(ious)
        iou_indicator = torch.as_tensor(iou_indicator)

        if device:
            ious = ious.to(device)
            iou_indicator = iou_indicator.to(device)

        return ious, iou_indicator

    # @classmethod
    # def unnormalize(self, tensor, mean, std):
    #     for t, m, s in zip(tensor, mean, std):
    #         t.mul_(s).add_(m)
    #         # The normalize code -> t.sub_(m).div_(s)
    #     return tensor

    @classmethod
    def pixel_accuracy(self, preds, label):
        valid = label >= 0
        acc_sum = (valid * (preds == label)).sum()
        valid_sum = valid.sum()
        acc = float(acc_sum) / (valid_sum + 1e-10)
        return acc, valid_sum

    @staticmethod
    def class_level_mask_iou_(a, b):
        # computing the mask IoU for each class

        class_mIoU = []

        assert a.shape == b.shape, "pred and label shape mismatch ... "

        for class_idx in range(a.shape[1]):
            each_class_a = a[:, class_idx, :, :]
            each_class_b = b[:, class_idx, :, :]

            isc = (each_class_a * each_class_b).sum()
            unn = (each_class_a + each_class_b).sum()
            z = unn - isc

            if z == 0:
                each_class_miou = 0
            else:
                each_class_miou = isc / z

            class_mIoU.append(each_class_miou)

        return class_mIoU

    @classmethod
    def dice_wobg_pytorch(
        self,
        pred,
        target,
        device=None,
        n_classes=12,
    ):
        dice_list = []
        dice_indicator = []
        pred = pred.view(-1)
        target = target.view(-1)

        for cls in range(1, n_classes):
            pred_inds = pred == cls
            target_inds = target == cls
            dice_intersection = (pred_inds[target_inds]).long().sum().item()
            dice_union = pred_inds.long().sum().item() + target_inds.long().sum().item()

            if dice_union == 0:
                dice_list.append(torch.tensor(0.0))
                dice_indicator.append(torch.tensor(0.0))
            else:
                dice_list.append(
                    torch.tensor(
                        float(2 * dice_intersection) / float(max(dice_union, 1))
                    )
                )
                dice_indicator.append(torch.tensor(1.0))

        dice_list = torch.as_tensor(dice_list)
        dice_indicator = torch.as_tensor(dice_indicator)

        if device:
            dice_list.to(device)
            dice_indicator.to(device)

        return dice_list, dice_indicator

    @classmethod
    def dice_wbg_pytorch(
        self,
        pred,
        target,
        device=None,
        n_classes=12,
    ):
        dice_list = []
        dice_indicator = []
        pred = pred.view(-1)
        target = target.view(-1)

        for cls in range(n_classes):
            pred_inds = pred == cls
            target_inds = target == cls
            dice_intersection = (pred_inds[target_inds]).long().sum().item()
            dice_union = pred_inds.long().sum().item() + target_inds.long().sum().item()

            if dice_union == 0:
                dice_list.append(torch.tensor(0.0))
                dice_indicator.append(torch.tensor(0.0))
            else:
                dice_list.append(
                    torch.tensor(
                        float(2 * dice_intersection) / float(max(dice_union, 1))
                    )
                )
                dice_indicator.append(torch.tensor(1.0))

        dice_list = torch.as_tensor(dice_list)
        dice_indicator = torch.as_tensor(dice_indicator)

        if device:
            dice_list.to(device)
            dice_indicator.to(device)

        return dice_list, dice_indicator


class Segmentation:
    @classmethod
    def generate_dict_single_batch(
        self,
        config,
        raw_image_tensor,
        prediction_tensor,
        gt_tensor,
    ):

        dict_single_batch = {}

        # We should make sure that the number of images in each row is not too large;
        if raw_image_tensor.shape[0] > 16:
            num_images_each_row = 16
        else:
            num_images_each_row = raw_image_tensor.shape[0]

        for image_idx in range(num_images_each_row):
            each_raw_image = UnNormalize.denorm(config, raw_image_tensor[image_idx])

            each_prediction = prediction_tensor.argmax(1)[image_idx]

            each_gt_mask = gt_tensor[image_idx]

            dict_single_batch["raw_image" + str(image_idx)] = each_raw_image.permute(
                1, 2, 0
            )
            # print("each_raw_image: {}".format(each_raw_image.shape))
            dict_single_batch["rawPrediction_overlay_" + str(image_idx)] = [
                each_raw_image.permute(1, 2, 0),
                each_prediction,
            ]
            dict_single_batch["prediction_" + str(image_idx)] = each_prediction
            dict_single_batch["gt_mask" + str(image_idx)] = each_gt_mask

        return dict_single_batch

    @classmethod
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

    @classmethod
    def decode_segmap_tocolor(self, temp, n_classes=19):

        label_colours = Segmentation.set_label_colours(n_classes)
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, n_classes):
            r[temp == l] = label_colours[l][0]
            g[temp == l] = label_colours[l][1]
            b[temp == l] = label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    @classmethod
    def set_label_colours(self, num_classes):

        colors = Segmentation.colormap(num_classes)

        label_colours = dict(zip(range(num_classes), colors))

        return label_colours


    @classmethod
    def plot2file_from_dict_list(
        self, config, dict_list, device, epoch_index, batch_index=None
    ):

        gpu_data_dict = {}
        gpu_data_dict[device] = dict_list

        # print("gpu_data_dict: {}".format(gpu_data_dict))

        for gpu_idx, gpu_dict_list in gpu_data_dict.items():
            print("{}, gpu_idx: {}".format(config["experiment_name"], gpu_idx))

            len_list = [len(dict_i) for dict_i in gpu_dict_list]

            # In each row, the number of images can be different.
            num_col = max(len_list)

            # Each key in the dict represents one image.
            num_row = len(gpu_dict_list)

            fig, axes = plt.subplots(
                num_row, num_col, figsize=(num_col * 5, num_row * 5)
            )

            for batch_idx in range(num_row):
                # print("batch_idx: {}".format(batch_idx))

                for image_idx, (image_key, each_image_dict_item) in enumerate(
                    gpu_dict_list[batch_idx].items()
                ):

                    # print("each_image_dict_item: {}".format(each_image_dict_item))
                    # print("image_idx: {}".format(image_idx))

                    image_key_list = str(image_key).split("_")

                    if num_col == 1:
                        if num_row == 1:
                            axes_plt = axes
                        else:
                            axes_plt = axes[batch_idx]
                    else:
                        if num_row == 1:
                            axes_plt = axes[image_idx]
                        else:
                            axes_plt = axes[batch_idx, image_idx]

                    if "raw" in image_key_list:

                        each_image_dict_item_raw = (
                            each_image_dict_item.detach().cpu().numpy()
                        )
                        if np.max(each_image_dict_item_raw) < 1.1:
                            axes_plt.imshow(
                                (each_image_dict_item_raw * 255).astype(np.uint8)
                            )
                        else:
                            axes_plt.imshow(each_image_dict_item_raw.astype(np.uint8))

                    elif "overlay" in image_key_list:

                        each_image_dict_item_raw = (
                            each_image_dict_item[0].detach().cpu().numpy()
                        )
                        each_image_dict_item_prediction = (
                            each_image_dict_item[1].detach().cpu().numpy()
                        )

                        if np.max(each_image_dict_item_raw) < 1.1:
                            axes_plt.imshow(
                                (each_image_dict_item_raw * 255).astype(np.uint8)
                            )
                        else:
                            axes_plt.imshow((each_image_dict_item_raw).astype(np.uint8))

                        prediction_mask = Segmentation.decode_segmap_tocolor(
                            each_image_dict_item_prediction, config["num_classes"]
                        )
                        axes_plt.imshow(prediction_mask, alpha=0.6)

                    elif "prediction" in image_key_list:

                        each_image_dict_item_prediction = (
                            each_image_dict_item.detach().cpu().numpy()
                        )
                        prediction_mask = Segmentation.decode_segmap_tocolor(
                            each_image_dict_item_prediction, config["num_classes"]
                        )
                        axes_plt.imshow(prediction_mask)

                    elif "gt" in image_key_list:

                        each_image_dict_item_gt = (
                            each_image_dict_item.detach().cpu().numpy()
                        )
                        gt_mask = Segmentation.decode_segmap_tocolor(
                            each_image_dict_item_gt, config["num_classes"]
                        )
                        axes_plt.imshow(gt_mask)

                    axes_plt.set_title(image_key, fontsize=9)
                    axes_plt.set_axis_off()

            os.makedirs(
                os.path.join(
                    config["output_folder"],
                    config["model_name"] + "_" + config["dataset_name"],
                ),
                exist_ok=True,
            )

            if batch_idx is None:
                plt.savefig(
                    os.path.join(
                        config["output_folder"],
                        config["model_name"] + "_" + config["dataset_name"],
                        "segemntation-"
                        + str(config["experiment_name"])
                        + "ep-"
                        + str(epoch_index)
                        + "-g-"
                        + str(gpu_idx)
                        + ".pdf",
                    )
                )
            else:
                plt.savefig(
                    os.path.join(
                        config["output_folder"],
                        config["model_name"] + "_" + config["dataset_name"],
                        "segemntation-"
                        + str(config["experiment_name"])
                        + "ep-"
                        + str(epoch_index)
                        + "-bs-"
                        + str(batch_index)
                        + "-g-"
                        + str(gpu_idx)
                        + ".pdf",
                    )
                )
            plt.close(fig)


if __name__ == "__main__":

    # Calculate Dice with Numpy;
    k = 1
    seg = np.zeros((100, 100), dtype="int")
    seg[30:70, 30:70] = k

    gt = np.zeros((100, 100), dtype="int")
    gt[30:70, 40:80] = k

    dice = np.sum(seg[gt == k]) * 2.0 / (np.sum(seg) + np.sum(gt))
    print("Dice similarity score is {}".format(dice))

    # Calclulate Dice with Pytorch
    seg_torch = torch.from_numpy(seg)
    gt_torch = torch.from_numpy(gt)

    dice_list, dice_indicator = SegmentationEvaluation.dice_without_bg(
        seg_torch, gt_torch, n_classes=2
    )
    print("dice_list: {}, dice_indicator: {}".format(dice_list, dice_indicator))
