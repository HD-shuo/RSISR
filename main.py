import argparse, os, sys, datetime, glob, importlib, csv

os.environ["HUGGINGFACE_HOME"] = "/share/program/dxs/huggingface"

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
from utils.model_utils import load_model_path_by_args
from data import DInterface
from model import MInterface


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
        filename='best-{epoch:02d}-{mpsnr:.2f}-{mssim:.3f}--{fid_score:.2f}--{lpips:.2f}',
        save_top_k=1,
        mode='max',
        save_last=True,
        dirpath='/home/work/daixingshuo/RSISR/checkpoint'
    ))

    if conf.model.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def main():
    configdir = "/home/work/daixingshuo/RSISR/configs/ddpm.yaml"
    conf = OmegaConf.load(configdir)
    seed = conf.other_params.seed
    pl.seed_everything(seed)
    load_path = load_model_path_by_args(conf.model)
    #for test pre_model
    load_path = None

    # data
    data_module = DInterface(**conf.data)
    
    # model
    if load_path is None:
        model = MInterface(**conf.model)
    else:
        model = MInterface(**conf.model)
        model.load_state_dict(torch.load(load_path), strict=False)
    
    # 加载预训练模型
    #pipeline = DiffusionPipeline.from_pretrained("/share/program/dxs/huggingface/stable-diffusion-xl-refiner-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    #pipeline.to("cuda")
    # 创建回调函数实例
    callbacks = load_callbacks(conf)
    # 创建logger
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_logger = TensorBoardLogger(save_dir='/share/program/dxs/RSISR/logs/train_logs', name=f"train_{current_time}")
    test_logger = TensorBoardLogger(save_dir='/share/program/dxs/RSISR/logs/test_logs', name=f"test_{current_time}")
    flag = conf.other_params.trainer_stage
    if flag == 'train':
        trainer = Trainer(callbacks=callbacks, logger=train_logger, **conf.trainer)
        #trainer.fit(pipeline, model, data_module)
        trainer.fit(model, data_module)
    elif flag == 'test':
        trainer = Trainer(logger=test_logger, **conf.trainer)
        trainer.test(model, data_module)
    else:
        raise ValueError("please specify the trainer mode")

if __name__ == "__main__":
    main()