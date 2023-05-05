from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import yaml

config_path = "/share/program/dxs/RSISR/configs/ptp.yaml"
"""
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
data = instantiate_from_config(config['data'])
"""
conf = OmegaConf.load(config_path)
data = instantiate_from_config(conf.data)
bs, base_lr = conf.data.params.batch_size, conf.model.base_learning_rate
print(bs, base_lr)
print(data.dataset)
print(data.batch_size)