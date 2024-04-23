import torch
import math
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class Swish(nn.Module):
    """
        desc:计算前向传播
        params:
            nn (_type_): _description_        
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
        desc:创建一个时间嵌入模块
        params:
            nn (_type_): _description_        
    """
    def __init__(self, T, d_model, dim):
        """
        初始化函数，用于实例化一个 Embedder 对象。
    
        Args:
            T (int): 序列的长度。
            d_model (int): 模型的维度，即输入和输出的维度。应为偶数。
            dim (int): 最后的线性变换前的维度。
    
        Raises:
            AssertionError: 如果 d_model 不是偶数。
        """
        assert d_model % 2 == 0
        super().__init__()
        # 生成一个步长为2的一维张量，长度为d_model,并把每个元素除以d_model再乘以log(10000)进行归一化。
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        # 计算指数函数，得到一个长度为d_model的张量。
        emb = torch.exp(-emb)
        # 创建一个长度为T的张量
        pos = torch.arange(T).float()
        # 将pos和emb进行乘法运算，得到一个形状为 [T, d_model // 2] 的张量 emb。这里使用了广播（broadcasting）机制，将 pos 扩展为二维张量，与 emb 进行逐元素相乘。
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        # 将emb中的每个元素分别应用正弦函数和余弦函数，然后将它们堆叠起来，形成一个新的张量。这样做是为了在时间嵌入中同时包含正弦和余弦的信息，以增强嵌入的表示能力。
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)
    
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        """
            desc:线性层的权重将以均匀分布的随机值初始化，而偏置参数将初始化为零值
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用 Xavier 均匀初始化方法初始化权重。
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb
    

class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(tdim, in_ch, out_ch, dropout, attn=False):
        super().__init__()
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear()
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()
    
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        # 通过 temb 经过 temb_proj 层进行处理，并将结果与 h 相加
        # 这里使用了广播操作 ([:, :, None, None])，将 temb 的维度扩展为与 h 相匹配。
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        # 计算残差连接
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UpSample(nn.Module):
    """
        desc:使用最邻近插值法对x进行上采样，2倍放大
        params:
            nn (_type_): _description_        
    """
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class DownSample(nn.Module):
    """
        desc:通过卷积层对x进行下采样，缩小一半
        params:
            nn (_type_): _description_        
    """
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


if __name__ == '__main__':
    T = 10
    d_model = 256
    dim = 384
    emb = TimeEmbedding(T, d_model, dim)
    print(emb)
