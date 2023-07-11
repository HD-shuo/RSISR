import torch
import pytorch_lightning as pl

# 导入你的模型类
from your_model import YourModel

class LightningTestModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # 创建你的模型实例并加载预训练的权重
        self.model = YourModel(...)
        self.model.load_state_dict(torch.load('path_to_pretrained_weights.pth'))

    def forward(self, x):
        # 定义前向传播逻辑
        return self.model(x)

    def test_step(self, batch, batch_idx):
        # 定义测试步骤逻辑
        x, y = batch
        y_pred = self(x)
        loss = ...  # 计算测试损失
        self.log('test_loss', loss)
        return loss

    def test_epoch_end(self, outputs):
        # 定义测试epoch结束时的操作
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_test_loss', avg_loss)

# 创建测试数据集或数据加载器
test_dataset = YourTestDataset(...)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

# 创建 Lightning 模块
model = LightningTestModule()

# 创建 Lightning Trainer
trainer = pl.Trainer()

# 进行测试
trainer.test(model, test_dataloaders=test_dataloader)

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
        conf.model.ckpt_path = load_path
    #print("model list:")
    #print(list(model.parameters()))
    #args.callbacks = load_callbacks(conf)
    # 创建回调函数实例
    callbacks = load_callbacks(conf)
    trainer = Trainer(callbacks=callbacks, **conf.trainer)
    trainer.fit(model, data_module)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)