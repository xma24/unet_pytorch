import numpy as np


def data_statistic(np_data, data_name=None):

    """
    np_data:
        - the data with numpy type;
    """

    data_min = np.min(np_data)
    data_max = np.max(np_data)
    data_mean = np.mean(np_data)
    data_type = type(np_data)

    if data_name is not None:
        print(
            "{}, min: {}, max: {}, mean: {}, shape: {}, data_type: {}".format(
                data_name, data_min, data_max, data_mean, np_data.shape, data_type
            )
        )
    else:
        print(
            "min: {}, max: {}, mean: {}, shape: {}, data_type: {}".format(
                data_min, data_max, data_mean, np_data.shape, data_type
            )
        )


import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold


class CustomizeDataset(torch.utils.data.Dataset):
    """
    config:
        - the configuration of the project
        - a dictionary
    data_label_dict:
        - a dictionary includes keys: data and labels.
    expr_phase:
        - "train", "val", "test" or "train_val:train", "train_val:val"
    transform:
        - data augmentation method
    """

    def __init__(self, config, data_label_dict, expr_phase, transform):

        self.config = config
        self.expr_phase = expr_phase

        self.transform = transform

        self.data = data_label_dict["data"]
        self.labels = data_label_dict["labels"]

        if (
            self.expr_phase == "train"
            or self.expr_phase == "val"
            or self.expr_phase == "test"
        ):
            self.sampled_data = self.data
            self.sampled_labels = self.labels
        else:

            expr_phase_list = self.expr_phase.split(":")

            if self.config["cross_validation"]:
                kfold = StratifiedKFold(
                    self.config["num_folds"], shuffle=True, random_state=69
                )
            elif self.config["hidden_validation"]:
                kfold = StratifiedKFold(self.config["num_folds"], shuffle=False)

            fold_idx = 0
            train_idx, val_idx = list(kfold.split(self.data, self.labels))[fold_idx]
            if expr_phase_list[-1] == "train":
                self.sampled_data, self.sampled_labels = (
                    self.data[train_idx],
                    self.labels[train_idx],
                )
            elif expr_phase_list[-1] == "val":
                self.sampled_data, self.sampled_labels = (
                    self.data[val_idx],
                    self.labels[val_idx],
                )

    def __len__(self):
        return len(self.sampled_data)

    def __getitem__(self, index):
        data_ret = self.sampled_data[index]
        labels_ret = self.sampled_labels[index]

        if not isinstance(data_ret, (dict)):
            data_ret = np.array(data_ret, dtype=np.float32)
            labels_ret = np.array(labels_ret, dtype=np.long)
        else:
            data_ret = data_ret
            labels_ret = np.array(labels_ret, dtype=np.long)

        # data_ret = self.transform(self.data[index])
        transformed = self.transform(image=data_ret)
        data_ret = transformed["image"]

        return data_ret, labels_ret


class UnNormalize(object):
    """
    config:
        - this is the config dictionary from yaml file, which should include mean and std.

    image:
        - image can be single image with 3 dim, or a batch of images with 4 dim.
    """

    @classmethod
    def denorm(self, config, image):

        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [CxHxW]"
            assert image.size(0) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip(image, config["mean"], config["std"]):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            # batch mode
            assert image.size(1) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip((0, 1, 2), config["mean"], config["std"]):
                image[:, t, :, :].mul_(s).add_(m)

        return image


import matplotlib
import numpy as np
import os
from matplotlib import cm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("ggplot")


class PlotFigures(object):
    @classmethod
    def plot_images_from_dict(self, image_dict_list):
        """
        image_dict_list:
            - image_dict_list is a list of dictionaries. each dict includes many images and the key is the image title.
            - [dict1, dict2, dict3, ...]
        """

        len_list = [len(dict_i) for dict_i in image_dict_list]

        num_col = max(len_list)
        num_row = len(image_dict_list)

        fig, axes = plt.subplots(num_row, num_col, figsize=(num_col * 5, num_row * 5))

        for image_dict_idx in range(num_row):

            for image_idx, (image_key, each_image) in enumerate(
                image_dict_list[image_dict_idx].items()
            ):

                image_key_list = str(image_key).split("_")

                if num_col == 1:
                    if num_row == 1:
                        axes_plt = axes
                    else:
                        axes_plt = axes[image_dict_idx]
                else:
                    if num_row == 1:
                        axes_plt = axes[image_idx]
                    else:
                        axes_plt = axes[image_dict_idx, image_idx]

                if np.max(each_image) < 1.1:
                    if "edge" in image_key_list:
                        axes_plt.imshow(
                            (each_image * 255).astype(np.uint8), cmap="gray"
                        )
                    else:
                        axes_plt.imshow((each_image * 255).astype(np.uint8))
                else:
                    if "edge" in image_key_list:
                        axes_plt.imshow(each_image.astype(np.uint8), cmap="gray")
                    else:
                        axes_plt.imshow(each_image.astype(np.uint8))
                axes_plt.set_title(image_key, fontsize=9)
                axes_plt.set_axis_off()

        return fig

    @classmethod
    def plot_scatter_from_dict(self, image_dict_list, cmap="tab10", s=0.5):
        """
        image_dict_list:
            - [dict1, dict2, ...]
            - dict1 = {"key": dict{"data": xxxx, "labels": xxxx}}
        cmap:
            - colormap; "tab10", "tab20", ...
        s:
            - the size of points
        """

        len_list = [len(dict_i) for dict_i in image_dict_list]

        num_col = max(len_list)
        num_row = len(image_dict_list)

        fig, axes = plt.subplots(num_row, num_col, figsize=(num_col * 5, num_row * 5))

        for image_dict_idx in range(num_row):

            for image_idx, (image_key, each_data_labels_dict) in enumerate(
                image_dict_list[image_dict_idx].items()
            ):

                # image_key_list = str(image_key).split("_")

                # to deal with only one line of images
                if num_col == 1:
                    if num_row == 1:
                        axes_plt = axes
                    else:
                        axes_plt = axes[image_dict_idx]
                else:
                    if num_row == 1:
                        axes_plt = axes[image_idx]
                    else:
                        axes_plt = axes[image_dict_idx, image_idx]

                axes_plt.scatter(
                    each_data_labels_dict["data"][:, 0],
                    each_data_labels_dict["data"][:, 1],
                    c=each_data_labels_dict["labels"],
                    cmap=cmap,
                    s=s,
                )

                axes_plt.set_title(image_key, fontsize=9)
                axes_plt.set_axis_off()

        return fig

    @classmethod
    def plot_from_dict_mask_overlay(self, image_dict_list):

        len_list = [len(dict_i) for dict_i in image_dict_list]

        num_col = max(len_list)
        num_row = len(image_dict_list)

        fig, axes = plt.subplots(num_row, num_col, figsize=(num_col * 5, num_row * 5))

        for image_dict_idx in range(num_row):

            for image_idx, (image_key, each_image_dict_item) in enumerate(
                image_dict_list[image_dict_idx].items()
            ):

                image_key_list = str(image_key).split("_")

                if num_col == 1:
                    if num_row == 1:
                        axes_plt = axes
                    else:
                        axes_plt = axes[image_dict_idx]
                else:
                    if num_row == 1:
                        axes_plt = axes[image_idx]
                    else:
                        axes_plt = axes[image_dict_idx, image_idx]

                if "merge" in image_key_list:
                    if np.max(each_image_dict_item[0]) < 1.1:
                        if "edge" in image_key_list:
                            axes_plt.imshow(
                                (each_image_dict_item[0] * 255).astype(np.uint8),
                                cmap="gray",
                            )
                        else:
                            axes_plt.imshow(
                                (each_image_dict_item[0] * 255).astype(np.uint8)
                            )
                            axes_plt.imshow(each_image_dict_item[1], alpha=0.6)
                    else:
                        if "edge" in image_key_list:
                            axes_plt.imshow(
                                each_image_dict_item.astype(np.uint8), cmap="gray"
                            )
                        else:
                            axes_plt.imshow(each_image_dict_item[0].astype(np.uint8))
                            axes_plt.imshow(each_image_dict_item[1], alpha=0.6)
                else:
                    each_image_to_plot = each_image_dict_item
                    if np.max(each_image_to_plot) < 1.1:
                        if "edge" in image_key_list:
                            axes_plt.imshow(
                                (each_image_to_plot * 255).astype(np.uint8), cmap="gray"
                            )
                        else:
                            axes_plt.imshow((each_image_to_plot * 255).astype(np.uint8))
                    else:
                        if "edge" in image_key_list:
                            axes_plt.imshow(
                                each_image_to_plot.astype(np.uint8), cmap="gray"
                            )
                        else:
                            axes_plt.imshow(each_image_to_plot.astype(np.uint8))
                axes_plt.set_title(image_key, fontsize=9)
                axes_plt.set_axis_off()

        return fig

    


if __name__ == "__main__":
    import numpy
    from PIL import Image

    image_dict_list = []
    for row_index in range(4):
        image_dict = {}
        for i in range(5):
            image_dict[i] = numpy.random.rand(32, 32, 3) * 255
        image_dict_list.append(image_dict)

    image_fig = PlotFigures.plot_images_from_dict(image_dict_list)
    plt.savefig("image_fig.pdf")
    plt.close(image_fig)

    image_dict_list = []
    for row_index in range(4):
        image_dict = {}
        for i in range(5):
            each_data_labels_dict = {}
            each_data_labels_dict["data"] = numpy.random.rand(1000, 2)
            each_data_labels_dict["labels"] = np.random.randint(10, size=1000)
            image_dict[i] = each_data_labels_dict
        image_dict_list.append(image_dict)

    image_fig = PlotFigures.plot_scatter_from_dict(image_dict_list, cmap="tab10", s=5)
    plt.savefig("scatter_image_fig.pdf")
    plt.close(image_fig)
