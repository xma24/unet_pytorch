import os
import subprocess
import zipfile

import yaml
from termcolor import cprint, colored

from dataset import *
from model import *
from utils import *


class downloadCheckpoint:
    @classmethod
    def download_checkpoint(self, fileId):

        cprint("Downloading checkpoint ...", "blue", "on_red")

        file_name = "./last.ckpt"

        fix_command0 = """; curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null;"""
        fix_command1 = """code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)";"""
        fix_command2 = """curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName};"""
        bash_command = (
            """fileId="""
            + fileId
            + """; fileName="""
            + file_name
            + fix_command0
            + fix_command1
            + fix_command2
        )

        if os.path.exists(file_name):
            cprint("Checkpoint already exists", "green")
        else:
            ret = subprocess.run(
                bash_command, shell=True, universal_newlines=True, check=True
            )
            print("ret: {}".format(ret))
            print("{}".format("Finish Downloading the Dataset ... "))

            if ret.returncode == 0:
                cprint("Downloading Successful", "green")
            else:
                cprint("Downloading Failed", "red")


class downloadDataset:
    def __init__(self, config):
        self.config = config

    def download_dataset(self, folder_clear=None):
        file_name = os.path.join(
            self.config["data_download_root"], self.config["dataset_name"] + ".zip"
        )

        fix_command0 = """; curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null;"""
        fix_command1 = """code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)";"""
        fix_command2 = """curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName};"""
        bash_command = (
            """fileId="""
            + self.config["fileId"]
            + """; fileName="""
            + file_name
            + fix_command0
            + fix_command1
            + fix_command2
        )

        if folder_clear is not None:
            os.system("rm -R " + self.config["data_download_root"])
            os.system("rm " + file_name)

        ## ****************************************************Downloading data ***************************************

        ## file already exists
        if os.path.exists(file_name):
            print("Data is ready")
            os.makedirs(self.config["data_download_root"], exist_ok=True)
            with zipfile.ZipFile(file_name, "r") as zip_ref:
                zip_ref.extractall(self.config["data_download_root"])
            print("{}".format("Finish Extracting the Dataset ... "))

        ## file not exists; download it
        else:
            print("{}".format("Downloading the Dataset ... "))
            # Trivial but horrible
            ret = subprocess.run(
                bash_command, shell=True, universal_newlines=True, check=True
            )
            print("ret: {}".format(ret))
            print("{}".format("Finish Downloading the Dataset ... "))

            if ret.returncode == 0:
                os.makedirs(self.config["data_download_root"], exist_ok=True)
                with zipfile.ZipFile(file_name, "r") as zip_ref:
                    zip_ref.extractall(self.config["data_download_root"])
                print("{}".format("Finish Extracting the Dataset ... "))
            else:
                print("Problem detected ... Data not extracted ...")


class PrepareDataset(object):
    """
    This class is mainly used to prepare datasets;
    1. determine whether to download the dataset;
    2. provide different dataset/dataloader class according to the dataset name;
    """

    def get_dataset_class(self, config):

        if config["download_dataset"]:
            dataset_download = downloadDataset(config)
            dataset_download.download_dataset()

        if config["dataset_name"] == "KITTI":
            dataset_class = KITTIDataset
        elif config["dataset_name"] == "VOC":
            dataset_class = PascalVOCDataset
            config["classes_names"] = PascalVOCDataset.CLASS_IDX_INV
        elif config["dataset_name"] == "MNIST":
            if config["model_name"] == "VCCModel":
                dataset_class = ClusteringDataloader
                assert (
                    config["validation"] == False
                ), "In Clustering mode, validation should be false .. "
            else:
                dataset_class = MNISTDataset
        elif config["dataset_name"] == "CIFAR10":
            if config["model_name"] == "VCCModel":
                dataset_class = ClusteringDataloader
                assert (
                    config["validation"] == False
                ), "In Clustering mode, validation should be false .. "
            else:
                dataset_class = CIFAR10Dataset
        elif config["dataset_name"] == "CarDamage" and not config["merged_labels"]:
            dataset_class = CarDamageClsSegDataset
            config["classes_names"] = CarDamageClsSegDataset.CLASS_IDX_INV
        elif config["dataset_name"] == "CarDamage" and config["merged_labels"]:
            dataset_class = CarDamageClsSegDataset
            config["classes_names"] = CarDamageClsSegDataset.CLASS_IDX_INV_UPDATE
        elif config["dataset_name"] == "Heatmap200K":
            dataset_class = Heatmap200KDataset
            config["classes_names"] = {0: "nodamage", 1: "damage"}
        elif config["dataset_name"] == "ImageNet":
            dataset_class = ImageNetDataset
        elif config["dataset_name"] == "corrMatStruct":
            dataset_class = CorrMatStruct
        elif config["dataset_name"] == "ADNIDataset":
            dataset_class = ADNIDataset
        elif config["dataset_name"] == "siim":
            dataset_class = SIIMDataset

        return dataset_class

    def show_batch(self, config, self_logger, data_loader):

        batch_image_list = []

        for each_batch in data_loader:
            each_batch_dict = {}

            if config["dataset_name"] == "VOC":
                images, labels, masks = each_batch
            elif config["dataset_name"] == "KITTI":
                images, masks = each_batch
            else:
                images, labels = each_batch

            print("images: {}".format(images.shape))

            for image_idx in range(images.shape[0]):

                each_image = images[image_idx]

                if config["mean"] is None or config["std"] is None:
                    to_plot_image = each_image
                else:
                    unorm = UnNormalize(mean=config["mean"], std=config["std"])
                    to_plot_image = unorm(each_image)

                each_batch_dict[image_idx] = (
                    to_plot_image.permute(1, 2, 0).detach().cpu().numpy()
                )

            batch_image_list.append(each_batch_dict)
            break

        fig = PlotFigures.plot_from_dict(batch_image_list)
        self_logger.log_image("Show_Batch", fig)


class ExprSetting(object):
    def __init__(self, experiment_name):

        self.experiment_name = experiment_name

        self.hyper_parameters()
        self.set_checkpoint_folder()
        self.Model = self.dynamic_models()

    def hyper_parameters(self):

        with open(self.experiment_name + ".yaml") as file:
            self.config = yaml.safe_load(file)

        os.makedirs(self.config["data_download_root"], exist_ok=True)
        os.makedirs(self.config["result_folder"], exist_ok=True)

        self.config["experiment_name"] = self.experiment_name.split("/")[-1]
        print(f"experiment_name {self.config['experiment_name']}")

    def set_checkpoint_folder(self):

        ignore_list = [
            "data_download_root",
            "result_folder",
            "transform",
            "data_path",
            "dataset_name",
            "model_name",
            "expr_index",
            "mean",
            "std",
            "fileId",
            "data_root",
            "split_root",
        ]

        os.makedirs(self.config["result_folder"], exist_ok=True)

        logger_name = (
            "pl-"
            + self.config["model_name"]
            + "-"
            + self.config["dataset_name"]
            + "-expr-"
            + str(self.config["expr_index"])
            + "-"
            + self.config["experiment_name"]
            + "-"
            + str(self.config["expr_tag"])
        )

        print("logger_name: {}".format(logger_name))
        self.config["logger_name"] = logger_name

        self.config["output_folder"] = os.path.join(
            self.config["result_folder"], self.config["logger_name"]
        )
        os.makedirs(self.config["output_folder"], exist_ok=True)

        self.config["checkpoint_folder"] = logger_name

    def checkpoint_setting(self):
        from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

        lr_logger = LearningRateMonitor(logging_interval="epoch")

        model_checkpoint = ModelCheckpoint(
            dirpath=os.path.join(
                self.config["result_folder"],
                self.config["checkpoint_folder"],
                "checkpoint",
            ),
            save_last=True,
            save_weights_only=False,
            save_top_k=5,
            monitor="val_loss",
            mode="min",
        )

        return lr_logger, model_checkpoint

    def earlystop_setting(self):
        # ## https://www.youtube.com/watch?v=vfB5Ax6ekHo
        from pytorch_lightning.callbacks import EarlyStopping

        early_stop = EarlyStopping(
            monitor="val_loss", patience=500, strict=False, verbose=False, mode="min"
        )
        return early_stop

    def clearml_start(self):
        pass

    def clearml_end(self):
        pass

    def neptune_start(self):

        from pytorch_lightning.loggers.neptune import NeptuneLogger

        neptune_logger = NeptuneLogger(
            # offline_mode = True,
            # api_key="ANONYMOUS",
            # project_name="MENET",
            # api_key="YOUR_NEPTUNE_API_KEY", project_name="YOUR_NEPTUNE_PROJECT_PATH"
            api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMTliNDE2OC03MDllLTRiOTktYjVkYS0xNWJlMDRmM2ZiODUifQ==",
            project_name="postech.sharing/SemSeg",
        )

        expr_tag = self.config["expr_tag"]
        neptune_logger.append_tags(
            [
                self.config["model_name"],
                self.config["dataset_name"],
                self.config["experiment_name"],
                expr_tag,
            ]
        )

        return neptune_logger

    def neptune_end(
        self,
        Model,
        model_checkpoint,
        neptune_logger,
        test_dataloader,
        inferenceFunc=None,
    ):

        for k in model_checkpoint.best_k_models.keys():
            model_name = "checkpoints/" + k.split("/")[-1]
            neptune_logger.experiment.log_artifact(k, model_name)

        neptune_logger.experiment.set_property(
            "best_model_score", model_checkpoint.best_model_score.tolist()
        )

        if not self.config["continue_training"]:
            # if "continue_training" is false, the checkpoint is generated during the training process;
            # if "continue_training" is true, the checkpoint is ready for inference;
            # config["checkpoint_folder"] is calculated in the program;
            checkpoint_path = os.path.join(
                self.config["result_folder"],
                "neptune_models",
                self.config["checkpoint_folder"],
                "checkpoint.ckpt",
            )
        else:
            checkpoint_path = os.path.join(
                self.config["result_folder"], "neptune_models", "checkpoint.ckpt"
            )

        if inferenceFunc is not None:
            inferenceFunc(Model, checkpoint_path, test_dataloader)

        # Stop Neptune logger at the end
        neptune_logger.experiment.stop()

    def dynamic_models(self):
        if self.config["model_name"] == "UNETModel":
            UniModel = UNETModel
        elif self.config["model_name"] == "DEEPLABV3SegModel":
            UniModel = DEEPLABV3SegModel
        elif self.config["model_name"] == "DEEPLABV3SegReduceModel":
            UniModel = DEEPLABV3SegReduceModel
        elif self.config["model_name"] == "ResNet18":
            UniModel = ResNet18
        elif self.config["model_name"] == "ResNet50":
            UniModel = ResNet50
        elif self.config["model_name"] == "SimCLR":
            UniModel = SimCLR
        elif self.config["model_name"] == "BiSeNetV2Model":
            UniModel = BiSeNetV2Model
        elif self.config["model_name"] == "MobileNetV2Model":
            UniModel = MobileNetV2Model
        elif self.config["model_name"] == "MobileNetV3SmallModel":
            UniModel = MobileNetV3SmallModel
        elif self.config["model_name"] == "FastSCNNModel":
            UniModel = FastSCNNModel
        elif self.config["model_name"] == "BiSeNetV2Heatmap200KTLModel":
            UniModel = BiSeNetV2Heatmap200KTLModel
        elif self.config["model_name"] == "BiSeNetV2CarDamageToDTLModel":
            UniModel = BiSeNetV2CarDamageToDTLModel
        elif self.config["model_name"] == "BiSeNetV2ASPPModel":
            UniModel = BiSeNetV2ASPPModel
        elif self.config["model_name"] == "BiSeNetV2TODModel":
            UniModel = BiSeNetV2TODModel
        elif self.config["model_name"] == "VCCModel":
            UniModel = VCCModel
        elif self.config["model_name"] == "MENETModel":
            UniModel = MENETModel
        elif self.config["model_name"] == "OneStageModel":
            UniModel = OneStageModel

        return UniModel


if __name__ == "__main__":
    checkpoint_id = "1lcPRKcfCCpz-qOmWZdul88naX_xtQW1S"
    downloadCheckpoint.download_checkpoint(checkpoint_id)
