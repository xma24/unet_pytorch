import os
import warnings

import yaml

import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.plugins import DDPPlugin
from rich import print
from termcolor import cprint, colored

from expr_setting import *
from utils import *

matplotlib.use("Agg")

plt.style.use("ggplot")

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    experiment_name = "config"

    experiment_setting = ExprSetting(experiment_name)

    prepare_dataset = PrepareDataset()

    experiment_config = experiment_setting.config

    # For Product Evaluation
    experiment_config["logger"] = "csv"
    experiment_config["gpus"] = 1
    experiment_config["part_results_saving"] = True
    experiment_config["max_epochs"] = 1
    experiment_config["batch_size"] = 5

    if isinstance(experiment_config["gpus"], int):
        num_gpus = experiment_config["gpus"]
    else:
        gpu_list = experiment_config["gpus"].split(",")
        num_gpus = len(gpu_list)

    Model = experiment_setting.dynamic_models()

    lr_logger, model_checkpoint = experiment_setting.checkpoint_setting()
    early_stop = experiment_setting.earlystop_setting()

    # Setup your own logger such as neptune or clearml;
    if experiment_config["logger"] == "neptune":
        own_logger = experiment_setting.neptune_start()
    elif experiment_config["logger"] == "csv":

        own_logger = CSVLogger(
            os.path.join(
                experiment_config["result_folder"],
                experiment_config["expr_tag"],
                "csv_logs",
            ),
            name=experiment_config["result_folder"] + experiment_config["dataset_name"],
        )
    else:
        own_logger = experiment_setting.neptune_start()

    dataset_class = prepare_dataset.get_dataset_class(experiment_config)

    if experiment_config["cross_validation"]:
        assert (
            experiment_config["hidden_validation"] == False
            and experiment_config["test"] == False
        ), print("Cannot use hidden validation and cross-validation at the same time")

        assert (
            experiment_config["provided_validation"] == False
            and experiment_config["test"] == False
        ), print("Cannot use provided validation and cross-validation at the same time")

        dataset_instance = dataset_class(experiment_config)
        trian_dataloader_list, val_dataloader_list = dataset_instance.get_dataloaders()
        print("trian_dataloader_list: {}".format(len(trian_dataloader_list)))
        print("val_dataloader_list: {}".format(len(val_dataloader_list)))

        for fold_index in range(experiment_config["num_folds"]):
            print(">>>>>>>>>>>>>> Fold Index: {} >>>>>>>>>>>>>>".format(fold_index))
            train_dataloader = trian_dataloader_list[fold_index]
            val_dataloader = val_dataloader_list[fold_index]
            print(
                "train_dataloader: {}, val_dataloader: {}".format(
                    len(train_dataloader), len(val_dataloader)
                )
            )

            model = Model(experiment_config, own_logger)

            if (len(train_dataloader) // num_gpus) // 6 >= 10:
                experiment_config["num_saved_batches"] = 10
            else:
                experiment_config["num_saved_batches"] = (
                    int((len(train_dataloader) // num_gpus) // 6) - 1
                )
            print("experiment_config: {}".format(experiment_config))

            trainer = pl.Trainer(
                gpus=experiment_config["gpus"],
                num_nodes=experiment_config["num_nodes"],
                precision=experiment_config["precision"],
                accelerator=experiment_config["accelerator"],
                logger=own_logger,
                callbacks=[lr_logger, model_checkpoint, early_stop],
                log_every_n_steps=1,
                # track_grad_norm=1,
                progress_bar_refresh_rate=experiment_config[
                    "progress_bar_refresh_rate"
                ],
                max_epochs=experiment_config["max_epochs"],
                # sync_batchnorm=True if num_gpus > 1 else False,
                plugins=DDPPlugin(find_unused_parameters=False),
            )

            trainer.fit(
                model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader
            )

    else:

        assert os.path.exists("checkpoint.yaml"), cprint(
            "checkpoint.yaml file cannot be found.", "red"
        )
        with open("checkpoint.yaml") as file:
            checkpoint_googledrive_id = yaml.safe_load(file)

        assert checkpoint_googledrive_id["checkpoint_id"] != "None", cprint(
            "Invalid Checkpoint ID.", "red"
        )
        downloadCheckpoint.download_checkpoint(
            checkpoint_googledrive_id["checkpoint_id"]
        )

        dataset_instance = dataset_class(
            experiment_config, validation=experiment_config["validation"]
        )
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

        model = Model(experiment_config, own_logger)

        # if (len(train_dataloader) // num_gpus) // 6 >= 10:
        #     experiment_config["num_saved_batches"] = 10
        # else:
        #     experiment_config["num_saved_batches"] = (
        #         int((len(train_dataloader) // num_gpus) // 6) - 1
        #     )

        experiment_config["num_saved_batches"] = 2

        assert os.path.exists("last.ckpt"), "Checkpoint is not found."

        model = Model.load_from_checkpoint(
            "last.ckpt", config=experiment_config, logger=own_logger
        )
        experiment_config["gpus"] = 1
        cprint("Model uses the pretrained weights.", "green")

        print("experiment_config: {}".format(experiment_config))

        trainer = pl.Trainer(
            gpus=experiment_config["gpus"],
            num_nodes=experiment_config["num_nodes"],
            precision=experiment_config["precision"],
            accelerator=experiment_config["accelerator"],
            logger=own_logger,
            callbacks=[lr_logger, model_checkpoint, early_stop],
            log_every_n_steps=1,
            # track_grad_norm=1,
            progress_bar_refresh_rate=experiment_config["progress_bar_refresh_rate"],
            max_epochs=experiment_config["max_epochs"],
            # resume_from_checkpoint=resume_from_checkpoint,
            # sync_batchnorm=True if num_gpus > 1 else False,
            plugins=DDPPlugin(find_unused_parameters=False),
        )

        # cprint("Evaluating on Train Dataset ...", "blue")
        # trainer.test(model, test_dataloaders=train_dataloader)

        cprint("Evaluating on Validation Dataset ...", "blue")
        trainer.test(model, test_dataloaders=val_dataloader)

        cprint("Evaluating on Test Dataset ...", "blue")
        trainer.test(model, test_dataloaders=test_dataloader)

        # experiment_setting.neptune_end(Model, model_checkpoint, own_logger,
        #                             test_dataloader)

# kill $(ps aux | grep main.py | grep -v grep | awk '{print $2}')
