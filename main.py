import argparse, os, sys, datetime, glob, importlib, csv
from pathlib import Path
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from datetime import datetime

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.parsing import AttributeDict
import pytorch_lightning.callbacks as plc
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.loggers import TensorBoardLogger

from utils.model_utils import load_model_path_by_args
from data import DInterface
from model import MInterface


def load_callbacks(conf):
    callbacks = []
    ckpt_path = '/share/program/dxs/RSISR/checkpoint'
    v_num = conf.model.load_v_num
    if v_num > -1:
        ckpt_path = str(Path(ckpt_path, f'version_{v_num}'))
    callbacks.append(plc.EarlyStopping(
        monitor='mpsnr',
        mode='max',
        patience=10,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='mpsnr',
        filename='best-{epoch:02d}-{mpsnr:.2f}-{mssim:.3f}--{fid_score:.2f}--{lpips:.2f}',
        save_top_k=1,
        mode='max',
        save_last=True,
        dirpath= ckpt_path
    ))

    if conf.model.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def main():
    configdir = "/share/program/dxs/RSISR/configs/cons.yaml"
    # configdir = "/share/program/dxs/RSISR/configs/vit-conf.yaml"
    conf = OmegaConf.load(configdir)
    seed = conf.other_params.seed
    pl.seed_everything(seed)
    load_path = load_model_path_by_args(conf.model)
    #for test pre_model
    #load_path = None

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
    train_logger = TensorBoardLogger(save_dir='log/train_logs', name=f"train_{current_time}_version_{conf.model.load_v_num}")
    test_logger = TensorBoardLogger(save_dir='log/test_logs', name=f"test_{current_time}_version_{conf.model.load_v_num}")
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