import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image
from datetime import datetime

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.parsing import AttributeDict
import pytorch_lightning.callbacks as plc
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.loggers import TensorBoardLogger

from ldm.util import instantiate_from_config
from utils import load_model_path_by_args
from data import DInterface
from model import MInterface


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser

def load_callbacks(conf):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='mpsnr',
        mode='max',
        patience=10,
        min_delta=0.01
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='mpsnr',
        filename='best-{epoch:02d}-{mpsnr:.2f}-{mssim:.3f}',
        save_top_k=1,
        mode='max',
        save_last=True,
        dirpath='/share/program/dxs/RSISR/checkpoint'
    ))

    if conf.model.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def main(args):
    configdir = "/share/program/dxs/RSISR/configs/ptp.yaml"
    conf = OmegaConf.load(configdir)
    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(conf.model)

    # data
    data_module = DInterface(**conf.data)
    
    # model
    if load_path is None:
        model = MInterface(**conf.model)
    else:
        model = MInterface(**conf.model)
        model.load_state_dict(torch.load(load_path), strict=False)
    #print("model list:")
    #print(list(model.parameters()))
    #args.callbacks = load_callbacks(conf)
    # 创建回调函数实例
    callbacks = load_callbacks(conf)
    # 创建logger
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_logger = TensorBoardLogger(save_dir='/share/program/dxs/RSISR/logs/train_logs', name=f"train_{current_time}")
    test_logger = TensorBoardLogger(save_dir='/share/program/dxs/RSISR/logs/test_logs', name=f"test_{current_time}")
    flag = conf.other_params.trainer_stage
    if flag == 'train':
        trainer = Trainer(callbacks=callbacks, logger=train_logger, **conf.trainer)
        trainer.fit(model, data_module)
    elif flag == 'test':
        trainer = Trainer(logger=test_logger, **conf.trainer)
        trainer.test(model, data_module)
    else:
        raise ValueError("please specify the trainer mode")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)