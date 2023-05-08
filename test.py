from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import yaml
"""
config_path = "/share/program/dxs/RSISR/configs/ptp.yaml"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
data = instantiate_from_config(config['data'])

conf = OmegaConf.load(config_path)
data = instantiate_from_config(conf.data)
bs, base_lr = conf.data.params.batch_size, conf.model.base_learning_rate
print(bs, base_lr)
print(data.dataset)
print(data.batch_size)
"""

import argparse
from pytorch_lightning import Trainer

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
            "-n",
            "--name",
            type=str,
            const=True,
            default="",
            nargs="?",
            help="postfix for logdir",
        )
    parser.add_argument(
            "-r",
            "--resume",
            type=str,
            const=True,
            default="",
            nargs="?",
            help="resume from logdir or checkpoint in logdir",
        )
    return parser

parser = get_parser()
opt, unknown = parser.parse_known_args()
print(opt)
print(unknown)